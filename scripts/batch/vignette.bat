@echo off

set "vignette_apply="
for %%x in (%~1) do (
	if /i "%%x"=="-a" set "vignette_apply=1"
	if /i "%%x"=="--apply" set "vignette_apply=1"
)

if defined vignette_apply (
	echo Applying vignetting...
) else (
	echo Correcting vignetting...
)

for /f "tokens=1* delims==" %%x in ('"set match_map: 2> nul"') do (
	for /f "tokens=*" %%f in ('"%env_python% scripts\python\resolve.py -n %in_dir% %%y"') do (
		call :vignette %%f %1
	)
)

for /f "tokens=*" %%f in ('"%env_python% scripts\python\resolve.py -n %in_dir% %equalize_list%"') do (
	call :vignette %%f %1
)


goto :end

:vignette
	set "black_level=0"
	set "vignette_centre="
	set "vignette_polynomials="
	
	for /f "tokens=1,2*" %%x in ('"%exiftool% -s -IFD0:BlackLevel -XMP:VignettingCenter -XMP:VignettingPolynomial "%original_in_dir:"=%\%~1""') do (
		if "%%x"=="BlackLevel" (
			set "black_levels=0"
			
			for %%b in (%%z) do (
				set /a "black_level+=%%b"
				set /a "black_levels+=1"
			)
			
			for /f "tokens=*" %%b in ('"echo print(!black_level! / float(!black_levels!)) | %env_python%"') do set "black_level=%%b"
		) else (
			if "%%x"=="VignettingCenter" (
				for %%v in (%%z) do set "vignette_centre=!vignette_centre! %%v"
			) else (
				if "%%x"=="VignettingPolynomial" (
					for %%v in (%%z) do set "vignette_polynomials=!vignette_polynomials! %%v"
				)
			)
		)
	)
	
	%env_python% scripts\python\vignette.py %~2 %in_dir% %1 %out_dir% !black_level! !vignette_centre! !vignette_polynomials!
exit /b

:end