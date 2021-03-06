# Generated by Django 2.1.5 on 2020-03-26 00:08

from django.db import migrations, models
import multiselectfield.db.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='InputData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('algorithms', multiselectfield.db.fields.MultiSelectField(choices=[('AR', 'Auto Regression'), ('ARIMA', 'ARIMA')], max_length=8)),
                ('cluster', models.BooleanField(default=True)),
                ('log', models.BooleanField(default=False)),
                ('graph', models.BooleanField(default=True)),
                ('deepLearning', models.BooleanField(default=False)),
                ('file', models.FileField(upload_to='input/')),
            ],
        ),
        migrations.CreateModel(
            name='OutputData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('forecast_file', models.FileField(upload_to='')),
                ('cluster_file', models.FileField(upload_to='')),
                ('log_file', models.FileField(upload_to='')),
            ],
        ),
    ]
