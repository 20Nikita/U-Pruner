def add_torch_fx_class_name(ClassName):
    # Добавление поддерживаемых классов в torch fx
    if ClassName is not None:
        s1 = '(m.__module__.startswith("torch.nn") or m.__module__.startswith("torch.ao.nn")'
        path_fix_torchfx = (
            "/usr/local/lib/python3.8/dist-packages/torch/fx/_symbolic_trace.py"
        )
        s2 = s1
        for class_name in ClassName:
            s2 += f' or m.__module__.startswith("{class_name}")'
        with open(path_fix_torchfx, "r") as f:
            old_data = f.read()
        new_data = old_data.replace(s1, s2)
        with open(path_fix_torchfx, "w") as f:
            f.write(new_data)
