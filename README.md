# generate-jets

This repo contains models (WGAN and VAE) trained on jet images downloaded from CERN (the home of the Large Hadron Collider), which are used to generate new jet image data. 

![jets](https://github.com/G-Kurup/generate-jets/assets/130678299/216eb2cb-1cba-4e53-b423-4ae09835f99b)

Jets, in particle physics, are clusters or narrow cones of high energy particles. They are formed from cascaded decays/hadronisation of preceding higher-energy particles in the reaction. In a particle collision, the immediate collision products in the very high energy regime obey perturbative physics, and is calculable. Particles carrying a color charge, i.e. quarks and gluons, cannot exist in free form because of quantum chromodynamics (QCD) confinement which only allows for colorless states. Thus, quarks and gluons quickly 'hadronise' and form showers of large numbers of particles in a stochastic process. In simulations, this process is replicated through Monte Carlo methods. It is to be noted that the definition of a jet depends on the jet clustering algorithm used. Jet physics is a complicated subject, and is an active area of research. Understanding them thoroughly could very well lead to the next big discovery in particle physics!

#### Contents

1. `helpers.py` -- Some helper functions
2. `GAN.py` -- Contains a Wasserstein Generative Adversarial Network (WGAN) model, with gradient penalty
3. `VAE.py` -- Contains a Variational Auto-Encoder model
4. `Train_GAN.py` -- Training the GAN
5. `Train_VAE.py` -- Training the VAE


This project was initiated during the '51st SLAC Summer Institute' in August 2023, at the Stanford Linear Accelerator Complex. The theme for the summer workshop was “Artificial Intelligence in Fundamental Physics”. 






