# Generated by Django 2.1.5 on 2020-04-02 20:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0002_auto_20200326_0009'),
    ]

    operations = [
        migrations.RenameField(
            model_name='inputdata',
            old_name='graph',
            new_name='forecast',
        ),
        migrations.RemoveField(
            model_name='inputdata',
            name='algorithms',
        ),
        migrations.RemoveField(
            model_name='inputdata',
            name='deepLearning',
        ),
        migrations.RemoveField(
            model_name='inputdata',
            name='log',
        ),
    ]
