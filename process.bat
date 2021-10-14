@echo off

setlocal EnableDelayedExpansion

call scripts\batch\configure.bat

call scripts\batch\vignette.bat

set "in_dir=%out_dir%"
call scripts\batch\standardize.bat

call scripts\batch\vignette.bat --apply

call scripts\batch\metadata.bat

endlocal EnableDelayedExpansion

echo Image processing complete!
pause