def parse_args(text):
    args = list()
    if text == '' or text == None:
        return args
    for arg in text.split(' '):
        last = (args[-1] if any(args) else None)
        if last != None and last.startswith('"'):
            last += ' ' + arg
            if last.endswith('"'):
                last = last.strip('"')
            args[-1] = last
        elif arg != '':
            args.append(arg)
    return args