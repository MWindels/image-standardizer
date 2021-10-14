@echo off

for /f "tokens=1* delims==" %%x in ('"set match_map: 2> nul"') do (
	for /f "tokens=1* delims=:" %%i in ("%%x") do (
		echo Matching images to %%~j...
		%env_python% scripts\python\standardize.py -r "%%~j" %in_dir% %%y %out_dir%
	)
)

if defined equalize_list (
	echo Equalizing images...
	%env_python% scripts\python\standardize.py %in_dir% %equalize_list% %out_dir%
)