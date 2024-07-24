import yaml


class QuotedString(str):
    pass


class FloatString(float):
    pass


yaml.add_representer(
    QuotedString,
    lambda dumper, data: dumper.represent_scalar(
        u'tag:yaml.org,2002:str', data, style='\''
    ),
)
yaml.add_representer(
    FloatString,
    lambda dumper, data: dumper.represent_scalar(
        u'tag:yaml.org,2002:str', f'{data:f}', style='\''
    ),
)
