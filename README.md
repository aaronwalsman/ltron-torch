# Learning to Build by Building Your Own Instructions
This branch is to track the code that was used to generate the experiments in ECCV'24 [Learning to Build by Building Your Own Instructions.](https://arxiv.org/abs/2410.01111)

```
@article{walsmanlearning,
  title={Learning to Build by Building Your Own Instructions},
  author={Walsman, Aaron and Zhang, Muru and Fishman, Adam and Farhadi, Ali and Fox, Dieter}
}
```

In the "Break and Make" problem, an agent attempts to fully disassemble a previously unseen LEGO model and then use its observations of this process to build it again from scratch.
![break and make](https://drive.google.com/uc?export=download&id=13kd-Wm_GZpJtgJc24kgwKqD6I_jMQXP4)

The model must use a 2D cursor-based action space to interact with the objects.  The agent first selects one of the discrete action categories below, then regresses heatmaps for where to click on the screen (the red boxes shown in the images below).

![action_space](https://drive.google.com/uc?export=download&id=1jn072PXEk-kqBGaMxwgm5F58JkMwc3lx)

To attack this problem, we built a new architecture called InstructioNet which builds a stack of instruction images for itself.  To implement this, we give our model a stack of images called the "Instruction Stack" and augment the action space above with an additional set of discrete actions that can either save the current observed image to the top of the instruction stack, or delete the top image of the instruction stack.

![architecture](https://drive.google.com/uc?export=download&id=1nqLRv425c2xp1dLbmzP854HMBPJ7M7rJ)

A gif of this in action can be seen below.  A highres version is available at [this link](https://drive.google.com/file/d/1kThh00ifMactuzyeldFCn72LSRKOhpEm/view?usp=drive_link).

![video](https://drive.google.com/uc?export=download&id=1Ib2gmR-oW011-sc1bHncY0B1s0eYNia1)

Below are several examples of success and failure cases:

![examples](https://drive.google.com/uc?export=download&id=1QwPK30ikRwJ_nRYRjoO5V9PAabNnq0ZI)
