# Fails unless the built executable carries its application manifest. Nothing
# else notices one missing: the app still starts, just DPI-unaware.
file(STRINGS "${EXE}" found REGEX "PerMonitorV2" LIMIT_COUNT 1)
if(NOT found)
    message(FATAL_ERROR "${EXE} has no PerMonitorV2 application manifest embedded")
endif()
