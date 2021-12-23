from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from model import Tacotron
from params_model import hparams
from params_train import *
from synthesisdataset import SynthesisDataset, SynthesisDataLoader
from visulization import plot_alignment_to_numpy, plot_spectrogram_to_numpy

def train_run(init_step=1):
    model = Tacotron(hparams)

    if model_checkout is not None:
        model.load_state_dict(torch.load(model_checkout))

    dataset = SynthesisDataset(Path(data_root))
    dataloader = SynthesisDataLoader(dataset, batch_size, sampler,
                                     batch_sampler, num_workers, pin_memory, timeout, worker_init_fn)

    model.train()
    model.zero_grad()
    if CUDA:
        model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    schedule = get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=num_warmup_steps,
                                               num_training_steps=total_step)
    all_loss = 0
    step = init_step
    for step, inputs in enumerate(dataloader, init_step):

        # precess data
        if CUDA:
            inputs.cuda()

        optimizer.zero_grad()

        model_outputs = model(inputs)
        loss = model.get_loss(model_outputs, [inputs.mels.transpose(1,2), inputs.gate_target], inputs.output_lengths)

        # return inputs, model_outputs
        loss /= batch_size
        all_loss += loss.item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_tresh)
        optimizer.step()
        schedule.step()

        if step % steps_per_show_loss == 0:
            alignments = model_outputs[3]
            print(f"step: {step}, loss : {all_loss/steps_per_show_loss:.4f}")
            all_loss = 0
            idx = int(torch.randint(batch_size, (1,)).item())
            spectro_length = int(inputs.output_lengths[idx].item())
            encode_length = int(inputs.text_lengths[idx].item())
            # print(alignments.shape, inputs.text_lengths)
            plot_alignment_to_numpy(alignments.detach().cpu().numpy()[idx, :spectro_length, :encode_length].T, info=f'step_{step}')
            plot_spectrogram_to_numpy(inputs.mels.detach().cpu().numpy()[idx, :spectro_length, :].T, model_outputs[1].detach().cpu().numpy()[idx, :, :spectro_length], info=f'step_{step}')

        if step % steps_per_checkout == 0:
            model.cpu()
            torch.save(model.state_dict(), checkout_dir + f'/{step}_tacotron.pth')
            if CUDA:
                model.cuda()