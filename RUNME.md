RUNME.md

Clone repo to location. Navigate to that location.

In Python, install ` pip install opencv-python`

How to build the docker container:

docker build -t zhengzheng/finalproject:latest .

How to run the docker container:

inlcude winpty if running on windows machine

winpty docker run -ti -v /${pwd.local_where_the_files_are}:/app zhengzheng/finalproject:latest
