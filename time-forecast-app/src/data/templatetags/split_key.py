from django import template

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