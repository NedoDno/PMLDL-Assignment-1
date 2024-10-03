## Build and Run
You can build and run the api and app using the following commands from the :
```commandline
cd code/deplotment/
docker-compose up --build
```

after that you can open http://localhost:8501 in browser and upload image of number to get prediction: that is the number on the image.

The model will work better if the image will be 28*28 px and if the number on the image will be white colored on black background. 