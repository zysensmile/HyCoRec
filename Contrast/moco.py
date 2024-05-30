import torch
from torch import nn
from loguru import logger


class MemoryMoCo(nn.Module):
    """Fixed-size queue with momentum encoder"""

    def __init__(self, input_size, output_size, queue_size, temperature=0.07, use_softmax=False):
        super(MemoryMoCo, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.queue_size = queue_size
        self.temperature = temperature
        self.use_softmax = use_softmax

        self.index = 0
        self.memory = nn.Parameter(torch.zeros((self.queue_size, self.input_size), dtype=torch.float))
        nn.init.normal_(self.memory, mean=0, std=self.input_size ** -0.5)

        logger.info(f"[Using queue shape: ({self.queue_size}), ({self.input_size})]")

    def forward(self, q, k, gpu):
        batch_size = q.shape[0]
        k = k.detach()

        # pos logit
        l_pos = torch.bmm(q.view(batch_size, 1, -1), k.view(batch_size, -1, 1))  # (32, 1, 64) * (32, 64, 1)
        l_pos = l_pos.view(batch_size, 1)  # (32, 1)
        # neg logit
        queue = self.memory.clone()
        l_neg = torch.mm(queue.detach(), q.transpose(1, 0))  # (16384, 64) * (64, 32)
        l_neg = l_neg.transpose(0, 1)  # (32, 16384)

        out = torch.cat((l_pos, l_neg), dim=1)  # (32, 16385)
        out = torch.div(out, self.temperature)
        out = out.squeeze().contiguous()

        # update memory
        with torch.no_grad():
            out_ids = torch.arange(batch_size).cuda(gpu)
            out_ids += self.index
            out_ids = torch.fmod(out_ids, self.queue_size)
            out_ids = out_ids.long()
            self.memory.index_copy_(0, out_ids, k)
            self.index = (self.index + batch_size) % self.queue_size

        return out