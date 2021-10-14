@echo off

for /f "tokens=* eol=;" %%l in (config.txt) do (
	set "%%l"
)

set "original_in_dir=%in_dir%"
set "original_out_dir=%out_dir%"
set "env_python="%env_dir:"=%\Scripts\python.exe""

if not exist %env_python% (
	%python% -m venv %env_dir%
	%env_python% -m pip install --upgrade pip
	%env_python% -m pip install -r requirements.txt
)