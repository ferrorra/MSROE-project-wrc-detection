## Processus du prof : 
### Conversion pour standardiser les mesures
-  saturated permeability (`k_sat`) from cm/day to m/s, particle size from µm to mm
- pressure head (`preshead`) from cm_H2O to kPa.

### Data Visualization
- **correlation heatmap**  soil properties et autre variables.
- **Plots porosity vs. permeability**
- **Plots all particle size distributions (GSD)** et water retention curves (WRC).
- **Focuses on sands**  filtering  soil texture ("sand") avec vis ge  GSD and WRC.
- **Interpolates GSD curves** select code et interpoler particle_size (cumulative percentages for 7 values), assurer consistence de comparasion.

### Feature Selection and Construction

- water retention curves : `preshead` (pressure head) and `theta` (volumetric water content)
- `bulk_density` (rho) influence on soil structure and water retention capacity.
- Porosity et permeability 'k_sat' are selected by teacher
- GSD is made by particle_size and fraction size.