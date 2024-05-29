import numpy as np
import torch
import matplotlib.pyplot as plt

## Filters out low energy pixels (physically, these would not trigger the detector) ##

def filter(img_batch, threshold = 1e-3):
    img_batch[img_batch < threshold] = 0
    return img_batch
    

## Prints a few random jet images from a batch ##

def show_jets(img_batch, rows= 2, columns= 3):

    fig = plt.figure(figsize=(11, 7))
    rng = np.random.default_rng(42)
    rand_nums = rng.integers(low=0, high=img_batch.shape[0], size=columns*rows)
    for i, num in enumerate(rand_nums):
        fig.add_subplot(rows, columns, i+1)
        img = img_batch[num].cpu().detach().numpy()
        img = img
        plt.imshow(img[0], origin='lower')
        plt.colorbar()
    plt.show()
    

## Plots a histogram of pixel values ##

def show_hist(resized_tensor_images, img_batch, bins=20):

    # Choose bins with real images
    min = torch.log(torch.min(resized_tensor_images[resized_tensor_images>0])).item()
    max = torch.log(torch.max(resized_tensor_images)).item()
    bin_width = (max-min)/bins
    x = [min + i*bin_width for i in range(bins)]

    img_batch = img_batch.cpu()
    hist = torch.histc(torch.log(img_batch[img_batch>0]), bins=bins, max=max, min=min)

    plt.bar(x, hist, align='edge', width=0.7)
    plt.xlabel("Log(Non-Zero Pixel Values)")
    plt.show()


## Cyclical annealing of KL factor ('beta') for VAE ##

def kl_factor(i, len_dataloader):
    peak = 0.9*len_dataloader
    if i < peak:
        return i/peak
    else:
        return 1.0
    

## Computes Gradient Penalty term for GAN ##

def gradient_penalty(netD, real_images, fake_images, batch_size, gamma=10, device="cuda:0"):

    eta = torch.rand(real_images.size(0), 1, 1, 1)
    eta = eta.expand(real_images.size(0), real_images.size(1), real_images.size(2), real_images.size(3))
    eta = eta.to(device)

    interpolated = eta * real_images + ((1 - eta) * fake_images)

    # calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                            create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    grad_penalty = torch.mean(gamma * ((gradients_norm - 1) ** 2))

    return grad_penalty
