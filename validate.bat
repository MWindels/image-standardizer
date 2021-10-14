@echo off

setlocal EnableDelayedExpansion

call scripts\batch\configure.bat

for /f "tokens=1* delims==" %%x in ('"set match_map: 2> nul"') do (
	for /f "tokens=*" %%f in ('"%env_python% scripts\python\resolve.py -n %out_dir% %%y"') do (
		call :validate %%f
	)
)

for /f "tokens=*" %%f in ('"%env_python% scripts\python\resolve.py -n %out_dir% %equalize_list%"') do (
	call :validate %%f
)

endlocal EnableDelayedExpansion

echo Validation complete!
pause


goto :end

:validate
	echo Validating metadata of %~1...
	
	set "in_total=0"
	for /f "tokens=1,2,3*" %%p in ('"%exiftool% -s -G1 --System:All "%original_in_dir:"=%\%~1" | sort"') do (
		set /a "in_total+=1"
		set "in_tags[!in_total!]=%%p %%q"
		set "in_values[!in_total!]=%%s"
	)
	
	set "out_total=0"
	for /f "tokens=1,2,3*" %%p in ('"%exiftool% -s -G1 --System:All "%out_dir:"=%\%~1" | sort"') do (
		set /a "out_total+=1"
		set "out_tags[!out_total!]=%%p %%q"
		set "out_values[!out_total!]=%%s"
	)
	
	set "in_index=1"
	set "out_index=1"
	set /a "grand_total=!in_total!+!out_total!"
	for /l %%x in (1, 1, !grand_total!) do (
		if !in_index! leq !in_total! (
			for %%i in (!in_index!) do (
				set "in_tag=!in_tags[%%i]!"
				set "in_value=!in_values[%%i]!"
			)
			
			if !out_index! leq !out_total! (
				for %%o in (!out_index!) do (
					set "out_tag=!out_tags[%%o]!"
					set "out_value=!out_values[%%o]!"
				)
				
				if "!in_tag!" leq "!out_tag!" (
					if "!in_tag!"=="!out_tag!" (
						if not "!in_value!"=="!out_value!" (
							echo =/=	!in_tag! !in_value! ^(!out_value!^)
						)
					) else (
						echo N/A	!in_tag! !in_value!
					)
					
					set /a "in_index+=1"
				)
				if "!in_tag!" geq "!out_tag!" (
					set /a "out_index+=1"
				)
			) else (
				echo N/A	!in_tag! !in_value!
				set /a "in_index+=1"
			)
		)
	)
exit /b

:end