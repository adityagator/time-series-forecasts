# Generated by Django 3.0.5 on 2020-04-26 04:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('data', '0012_inputdata_email'),
    ]

    operations = [
        migrations.AlterField(
            model_name='inputdata',
            name='email',
            field=models.BooleanField(default=False),
        ),
        migrations.CreateModel(
            name='UserHistory',
            fields=[
                ('user', models.TextField(default='dummy')),
                ('input', models.OneToOneField(default=0, on_delete=django.db.models.deletion.CASCADE, primary_key=True, serialize=False, to='data.InputData')),
                ('output', models.OneToOneField(default=0, on_delete=django.db.models.deletion.CASCADE, to='data.OutputData')),
            ],
        ),
    ]