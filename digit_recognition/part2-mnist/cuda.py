import torch

do_i_have_cuda = torch.cuda.is_available()

if do_i_have_cuda:
    print('Using fancy GPUs')
    # One way
    a = a.cuda()
    a = a.cpu()

    # Another way
    device = torch.device('cuda')
    a = a.to(device)

    device = torch.device('cpu')
    a = a.to(device)
else:
    print('CPU it is!')