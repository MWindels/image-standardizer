@echo off

echo Copying metadata...

set "out_dir_remainder= %out_dir:"=%"
set "out_dir_length=0"

:count_out_dir_length
for /f "tokens=1* delims=/\" %%x in ("!out_dir_remainder!") do (
	set "out_dir_remainder=%%y"
	set /a "out_dir_length+=1"
	goto :count_out_dir_length
)

%exiftool% -recurse -overwrite_original -tagsFromFile "%original_in_dir:"=%\%%:!out_dir_length!D\%%F" -All:All -XMP -IFD0:SubfileType -IFD0:SamplesPerPixel -IFD0:BlackLevelRepeatDim -IFD0:BlackLevel %out_dir%