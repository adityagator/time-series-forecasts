# Forecasting time series data
###### Authored by Aditya Bharadwaj (adityagator)

## Steps to run the code

#### clone repository
git clone https://github.com/adityagator/time-series-forecasts.git

#### create virtual environment
python3 -m venv ./env

#### activate environment
source env/bin/activate

#### change directory to time-forecast-app
cd time-forecast-app

#### install packages
cat requirements.txt | xargs -n 1 pip install

#### install packages which threw an error during installation seperately

#### create config.json file
{
        "SECRET_KEY": "",
        "EMAIL_HOST_USER": "",
        "EMAIL_HOST_PASSWORD": ""
}

#### Migrate Database
python3 manage.py migrate

#### Create Super Admin
python3 manage.py createsuperuser

#### run command
python3 manage.py runserver



Multiple Machine Learning algorithms including Neural Networks are used to forecast the demand of products 
to help organizations plan their warehousing needs.
