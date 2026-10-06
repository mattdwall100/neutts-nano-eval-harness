# Take-home: benchmark on-device speech, then make it better

## The idea

Our open-source NeuTTS models run text-to-speech entirely on-device. To make them better, we first need to know how well they actually perform, and to trust those numbers.

**Your task has two parts:**
1. Build a benchmarking framework for NeuTTS-Nano.
2. Use it to optimise NeuTTS-Nano as far as you can on your own machine.

## What we're asking for

### 1. A benchmarking framework
Something that measures how NeuTTS-Nano performs on-device, gives numbers you'd trust, and that the team could keep using.

You decide:
- what "performance" means here;
- what to measure;
- how to measure it.

We can run it ourselves with **one command**.

### 2. Optimise it
Optimise NeuTTS-Nano as far as you can on your own machine. What you optimise for is up to you.

We want you to go all out here. Don't stop at the easy wins: push it as far as you possibly can, and show us how far you got. Use your framework to show the results.

### 3. Design decisions
A markdown file in your repo (for example `DESIGN.md`) explaining all of your design decisions and why you made them.

## Practical details

- NeuTTS: https://github.com/neuphonic/neutts
- Everything runs locally on your own laptop. Tell us your hardware.
- Use whatever language you're most comfortable with. Python is completely fine. We're not expecting you to work in a language you're unfamiliar with.

## What to send back

A Git repository (or zip), with a README covering setup, how to run the benchmark, and your hardware, plus your design decisions file.


Any questions feel free to email!
