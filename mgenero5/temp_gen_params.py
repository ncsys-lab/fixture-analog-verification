sel_str = ''
for i in range(16):
    sel_str += f'        thm_sel_bld[{i}]: {1/16}\n'


text = '''
test1:
    gain:
    - coef:
SEL
      mode:
        dummy_digitalmode: 0
    offset:
    - coef:
        (1.0): 0.0000001
      mode:
        dummy_digitalmode: 0
'''
print(text.replace('SEL', sel_str))
