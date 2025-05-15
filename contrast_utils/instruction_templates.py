import re


_INSTRUCTION_TEMPLATES = [
    r"pick (.+)",
    r"move (.+) near (.+)",
    r"open (.+)",
    r"close (.+)",
    r"place (.+) into (.+)",
    r"put (.+) into (.+)",
    r"put (.+) on (.+)",
    r"stack (.+) on (.+)",
]

_ALL_DRAWER_PARTS = ['body', 'top drawer', 'middle drawer', 'bottom drawer']

def get_all_parts_of_drawer(objects):
    # if any part in objs, add all parts into objs
    is_any_in = False
    for part in _ALL_DRAWER_PARTS:
        if part in objects:
            is_any_in = True
            break
    
    if is_any_in:
        for part in _ALL_DRAWER_PARTS:
            if part not in objects:
                objects.append(part)
    
    return objects

def get_objects_from_instruction(instruction, get_all_parts=False):
    instruction = instruction.replace('the ', '') \
                             .replace('plate', 'plate larger') \
                             .replace('towel', 'table cloth shorter') \
                             .replace('block', 'cube')
    for template in _INSTRUCTION_TEMPLATES:
        match = re.match(template, instruction)
        if match:
            objects = list(match.groups())
            if get_all_parts:
                objects = get_all_parts_of_drawer(objects)
            return [obj.strip() for obj in objects]
    raise ValueError(f"Instruction '{instruction}' does not match any template")
