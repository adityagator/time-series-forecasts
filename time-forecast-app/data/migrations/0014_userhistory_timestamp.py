# Generated by Django 3.0.5 on 2020-04-26 04:47

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0013_auto_20200426_0408'),
    ]

    operations = [
        migrations.AddField(
            model_name='userhistory',
            name='timestamp',
            field=models.DateTimeField(default=datetime.datetime(2020, 4, 26, 4, 47, 16, 119557)),
        ),
    ]