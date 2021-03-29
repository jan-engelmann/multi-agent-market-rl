## Spinning Up
### Setup
- gym mujoco needs mjpro150 not mujoco200
- probably super hacky but `sudo ln -s /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/* /usr/local/include/` or maybe `export CPATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include` fixes wrong compiler versions
- check answers here: [stackoverflow answers](https://stackoverflow.com/questions/58278260/cant-compile-a-c-program-on-a-mac-after-upgrading-to-catalina-10-15/58349403#58349403)
- this fixed the unidentified develper problem: [link](https://osxdaily.com/2016/09/27/allow-apps-from-anywhere-macos-gatekeeper/)
