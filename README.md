# Machine Learning for Industry Project

## Machine Learning pipeline
In order to run the ML pipeline, open this folder in your terminal and run the file main.py.
Depending on your Python installation you may want to use `python main.py` or `python3 main.py`.

### Dependencies
If you don't already have Python installed on your computer, you can install it by following the instructions on https://www.python.org/downloads/.

The list of the required dependencies was created automatically by Pycharm and can be found in the file requirements.txt.
We expect it to be complete, but it's sometimes possible that a few dependencies are not recognised automatically.
In case some dependencies are missing, it is possible to install them through the command `pip install [dependency name]` or `pip3 install [dependency name]`, depending on your installation of python.


## Deployment
To deploy the web app use the command `cd deployment` to open the deployment folder and use `flask run` to start the web server.
You can then open http://127.0.0.1:5000 on your browser to visualize the web app.

The temporary credentials for admin login are:
- Username: admin
- Password: admin

### Debug
To deploy the app in debug mode, use the command `cd deployment` to open the deployment folder and use `flask --app app.py --debug run` to start the web server in debug mode.
You can then follow the instructions described above to interact with the web app.
Debug mode allows you to modify files and update the web app in real time, in addition with more descriptive logs.

### Web server
In case you don't want to run the app locally on your device, you can test it on our Hugging Face web server on https://huggingface.co/spaces/alphaaureus/ML4I.

### Known issues
It is necessary to logout before stopping the app, otherwise it might deny access when the app runs again on the same browser session.