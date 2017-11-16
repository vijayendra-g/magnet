# magnet  ToDo


How to combine splited files in windows

cd into yolo-data
copy /b file1 + file2 + file3 + file4 filetogether.tar.gz
uncompress filetogether.tar.gz


How to combine splited files in Ubuntu
cd into yolo-data
cat final-data.tar.gz* | tar xz


How to  split  big files in Ubuntu
tar cvzf - final-data | split -b 10m - final-data.tar.gz