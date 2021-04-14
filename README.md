# Ark
Musings on a modern C++ game engine architecture

## Building
The project uses Google's [Bazel](https://bazel.build/).
To install, merely copy the executable file to somewhere in your path.
Right now, there are only some tests written using the Google Test framework.
To build and run the tests, execute the following command:
```
bazel test //Ark/Tests:ArkTests
```
