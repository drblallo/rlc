import site
# Get the list of directories
site_packages_dirs = site.getsitepackages()

# Find the "site-packages" directory in the list
for dir in site_packages_dirs:
    if dir.endswith("site-packages"):
        target_dir = dir
        break
    else:
        target_dir=None
print(target_dir)

