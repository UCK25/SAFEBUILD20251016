import os, importlib, traceback
os.chdir(r"C:\Users\kenka\Downloads\SAFEBUILD20251001")
print('CWD', os.getcwd())
mods = ['database','main_window','camera_widget','observer']
for m in mods:
    try:
        mod = importlib.import_module(m)
        importlib.reload(mod)
        print('Imported', m)
    except Exception as e:
        print('Error importing', m, '->', e)
        traceback.print_exc()
print('Done')
