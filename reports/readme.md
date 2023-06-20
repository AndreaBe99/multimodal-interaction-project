# PlantUML

Install PlantUML with for Python with:

```cli
pip install plantuml
```

The PlantUML documentation can be found at [PlantUML Documentation](https://plantuml.com/).

To obtain images from PlantUML, use the following command:

```cli
python -m plantuml filename.txt -o path/to/output 
```

Example, execute the following command in `reports` folder to obtain the image from the file `uml_classes.txt` and save it in the folder `figures/`:

```cli
python -m plantuml uml_classes.txt -o figures/
```


