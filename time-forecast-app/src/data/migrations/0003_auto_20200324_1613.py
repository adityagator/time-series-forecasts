# Generated by Django 2.1.5 on 2020-03-24 16:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0002_inputdata_deeplearning'),
    ]

    operations = [
        migrations.AlterField(
            model_name='inputdata',
            name='cluster',
            field=models.BooleanField(default=True),
        ),
        migrations.AlterField(
            model_name='inputdata',
            name='deepLearning',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='inputdata',
            name='graph',
            field=models.BooleanField(default=True),
        ),
        migrations.AlterField(
            model_name='inputdata',
            name='log',
            field=models.BooleanField(default=False),
        ),
    ]
