# low-ban
Low bandwidth video experiment using dlib to implement face landmarking and an autoencoder.

During a hackathon at Microsoft, I tried to invent a super low-bandwidth video codec optimized for 2G networks. It garnered significant attention during the event, though it remained an experimental prototype. The core idea was to use facial landmarking to highlight the user's expressions, preserving non-verbal communication even when the video frame was highly compressed and blurry.

This repository represents the initial concept before I fully integrated an autoencoder-based codec. I plan to add the autoencoder code once I can simplify its dependencies. This repository includes the version of dlib that I used at the time.
![image](https://github.com/ZacWalk/low-ban/assets/181544/01160e1c-c3ce-40d4-895e-375e01c712d1)
