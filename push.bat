@echo off

if not exist Version.txt >Version.txt echo 0
for /f %%x in (Version.txt) do (
set /a ver=%%x+1
)
echo %ver% > Version.txt 
echo Dit is versie %ver%
