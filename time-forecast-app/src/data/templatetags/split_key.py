from django import template
from data.models import UserHistory
import datetime

register = template.Library()

@register.filter
def split_key(value):
    ship, prod, part = value.split("^")
    label = "ship pt: " + ship + "     " + "product hierarchy: " + prod + "     " + "part number: " + part
    return label

@register.filter
def remove_spaces(value):
    return value.replace(" ", "@")

@register.filter
def add_spaces(value):
    return value.replace("@", " ")

@register.filter
def handle_query(value, text):
    return value.handle_query(text)

@register.filter
def save_user_history(username, input, output):
    user_history_obj = UserHistory()
    user_history_obj.user = username
    user_history_obj.input = input
    user_history_obj.output = output
    user_history_obj.timestamp = datetime.now()
    user_history_obj.save()