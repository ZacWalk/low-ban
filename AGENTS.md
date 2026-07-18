This workspace contains the `low-ban` project, an experimental Windows application designed for exploring low bandwidth video communication via face landmarking and autoencoders.

# Technologies and Frameworks
* **Language:** C++17
* **Framework:** Win32 API
* **Multimedia Library:** Windows Media Foundation for webcam capture and frame handling
* **Machine Learning Library:** `dlib` for facial detection, shape prediction, and training pipelines

# Coding Guidelines
* Respect existing coding style: use standard C++ data structures (`std::vector`, `std::map`, etc.) and basic smart pointers.
* Do not alter the `dlib` library source code inside the `dlib/` folder. Apply changes outside unless strictly required.
* Ensure UI and non-UI threads sync properly using `std::mutex` or lock-free atomics when sending processed frames down to the renderer.
* When working with Win32 APIs, prefer error-checking using `SUCCEEDED`/`FAILED` macros on `HRESULT` return types where applicable.
* Build configuration targets `Debug|x64` using `MSBuild`.
