import neurokit2 as nk
import inspect

names = [name for name in dir(nk) if 'eeg' in name.lower() or 'artifact' in name.lower() or 'clean' in name.lower()]
print('matches:', names)
for fname in ['eeg_process','eeg_clean','eeg_clean_epochs','signal_clean','bio_process','ecg_clean']:
    print(fname, 'in nk?', hasattr(nk, fname))
    if hasattr(nk, fname):
        try:
            print('sig:', inspect.signature(getattr(nk, fname)))
        except Exception as e:
            print('sig error', e)

# try to import cleaning module
try:
    import neurokit2.nk as nk_internal
    print('nk_internal attrs:', [a for a in dir(nk_internal) if 'eeg' in a or 'clean' in a or 'artifact' in a])
except Exception:
    pass

# Check for functions in nk.signal
try:
    from neurokit2 import signal
    sig_names = [n for n in dir(signal) if 'clean' in n or 'artifact' in n or 'eeg' in n]
    print('neurokit2.signal matches:', sig_names)
except Exception:
    pass

# Look for high-level eeg functions
for module_name in ['eeg', 'ecg', 'signal', 'bio']:
    try:
        mod = getattr(nk, module_name)
        print(f"module {module_name} functions: ", [n for n in dir(mod) if 'clean' in n or 'process' in n or 'artifact' in n][:20])
    except Exception:
        pass

print('done')
