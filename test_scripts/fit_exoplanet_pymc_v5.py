import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pymc_ext as pmx
import exoplanet as xo
from astropy.io import fits
from astropy.table import Table
import arviz as az

# --- 1. CONFIGURACIÓN ---
FITS_FILE = "lightcurve_data_TOI-4948_V_CLEAN.fits"

# DATOS ESTELARES Y PLANETARIOS (ExoFOP / TESS)
R_STAR_MU    = 1.10267
R_STAR_SIGMA = 0.05    
M_STAR_MU    = 1.05
M_STAR_SIGMA = 0.05
PERIOD_VAL   = 0.4722347 

# CONFIGURACIÓN DE LIMPIEZA
SIGMA_SKY_CLIP = 3.0  # Eliminar puntos con cielo > Mediana + 3*Sigma

print(f"📖 Leyendo {FITS_FILE}...")
try:
    tbl = Table.read(FITS_FILE)
except FileNotFoundError:
    print(f"❌ Error: No encuentro el archivo {FITS_FILE}.")
    exit()

# --- 1.1 FILTRADO DE CALIDAD (NUBES) ---
# Extraemos datos crudos
raw_sky = tbl['SKY_MEAN'].value
raw_mag = tbl['MAG_CAL'].value
raw_mjd = tbl['MJD'].value
raw_err = tbl['MAG_ERR'].value

# Calculamos estadística del cielo
med_sky = np.median(raw_sky)
std_sky = np.std(raw_sky)
threshold_sky = med_sky + (SIGMA_SKY_CLIP * std_sky)

# Máscara: Solo mantenemos datos donde el cielo es estable
mask_good = raw_sky < threshold_sky
n_removed = len(raw_sky) - np.sum(mask_good)

print(f"🧹 FILTRO APLICADO: Eliminados {n_removed} puntos por cielo alto (> {threshold_sky:.1f} ADU)")

# Aplicamos máscara
t_mjd = raw_mjd[mask_good]
mag_cal = raw_mag[mask_good]
mag_err = raw_err[mask_good]

# Preparar tiempos
t_offset = t_mjd[0] 
t = t_mjd - t_offset 

# Flujo relativo normalizado
median_mag = np.median(mag_cal)
flux = 10**(-0.4 * (mag_cal - median_mag))
flux_err = flux * 0.921 * mag_err

# Guess inicial del centro
t0_guess = t[np.argmin(flux)]

print(f"⭐ Configuración Estelar: R={R_STAR_MU} R_sun, M={M_STAR_MU} M_sun")

# --- 2. MODELO BAYESIANO ---
print("⚙️ Construyendo modelo PyMC...")

with pm.Model() as model:
    # A) Priors Estelares
    r_star = pm.Normal("r_star", mu=R_STAR_MU, sigma=R_STAR_SIGMA)
    m_star = pm.Normal("m_star", mu=M_STAR_MU, sigma=M_STAR_SIGMA)

    # B) Priors Orbitales
    t0 = pm.Normal("t0", mu=t0_guess, sigma=0.05)
    period = pm.Deterministic("period", pm.math.constant(PERIOD_VAL))

    # C) Priors Planetarios
    log_r = pm.Normal("log_r", mu=np.log(0.08), sigma=0.5) 
    r = pm.Deterministic("r", pm.math.exp(log_r)) 
    
    # Parámetro de impacto (seguro para PyMC v5)
    b = pm.Uniform("b", lower=0, upper=2.0)
    
    # Oscurecimiento del Limbo
    u = xo.distributions.QuadLimbDark("u")
    mean = pm.Normal("mean", mu=1.0, sigma=0.01)

    # D) Órbita y Curva de Luz
    orbit = xo.orbits.KeplerianOrbit(
        period=period, t0=t0, b=b, r_star=r_star, m_star=m_star
    )
    light_curves = xo.LimbDarkLightCurve(u[0], u[1]).get_light_curve(
        orbit=orbit, r=r, t=t
    )
    mu_model = mean + pm.math.sum(light_curves, axis=-1)

    # F) Likelihood
    pm.Normal("obs", mu=mu_model, sigma=flux_err, observed=flux)

    # --- 3. OPTIMIZACIÓN (MAP) ---
    print("🚀 Buscando solución óptima (MAP)...")
    map_soln = pmx.optimize(start=model.initial_point())

    # --- 4. RESULTADOS ---
    r_prs_fit = map_soln['r']
    r_star_fit = map_soln['r_star']
    
    depth_percent = (r_prs_fit**2) * 100
    r_planet_rearth = r_prs_fit * r_star_fit * 109.076 
    t0_mjd = map_soln['t0'] + t_offset

    print("-" * 50)
    print(f"📊 RESULTADOS FINALES (FILTRADOS) PARA TOI-4948")
    print("-" * 50)
    print(f"Profundidad (Depth)   : {depth_percent:.3f} %")
    print(f"Radio Planeta Físico  : {r_planet_rearth:.2f} R_tierra")
    print(f"Parámetro Impacto (b) : {map_soln['b']:.3f}")
    print("-" * 50)

    # --- 5. GRAFICAR ---
    plt.switch_backend('Agg') 
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Modelo suave
    t_smooth = np.linspace(t.min(), t.max(), 1000)
    with model:
        orbit_map = xo.orbits.KeplerianOrbit(
            period=map_soln["period"], t0=map_soln["t0"], b=map_soln["b"],
            r_star=map_soln["r_star"], m_star=map_soln["m_star"]
        )
        lc_smooth = xo.LimbDarkLightCurve(map_soln["u"][0], map_soln["u"][1]).get_light_curve(
            orbit=orbit_map, r=map_soln["r"], t=t_smooth
        )
        model_smooth = map_soln["mean"] + np.sum(lc_smooth.eval(), axis=-1)
        
        # Modelo para residuos
        lc_points = xo.LimbDarkLightCurve(map_soln["u"][0], map_soln["u"][1]).get_light_curve(
            orbit=orbit_map, r=map_soln["r"], t=t
        )
        model_points = map_soln["mean"] + np.sum(lc_points.eval(), axis=-1)

    t_hours = t * 24.0
    t_smooth_hours = t_smooth * 24.0

    # Panel Superior
    ax1.errorbar(t_hours, flux, yerr=flux_err, fmt=".k", alpha=0.3, label="Datos Limpios")
    ax1.plot(t_smooth_hours, model_smooth, color="#d62728", lw=2, label="Modelo Exoplanet")
    
    # --- ZOOM INTELIGENTE (Percentiles) ---
    # Ignoramos outliers extremos para el escalado del eje Y
    y_low = np.percentile(flux, 0.5) 
    y_high = np.percentile(flux, 99.5)
    margin = (y_high - y_low) * 0.2
    ax1.set_ylim(y_low - margin, y_high + margin)

    stats_text = (rf"$R_p$ = {r_planet_rearth:.1f} $R_{{\oplus}}$" + "\n"
                  f"Depth = {depth_percent:.2f}%\n"
                  f"$b$ = {map_soln['b']:.2f}")
    
    ax1.text(0.02, 0.05, stats_text, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

    ax1.set_ylabel("Flujo Relativo Normalizado")
    ax1.set_title(f"Ajuste Final: TOI-4948 (MJD {t0_mjd:.2f})")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    # Panel Inferior
    resid = flux - model_points
    ax2.errorbar(t_hours, resid, yerr=flux_err, fmt=".k", alpha=0.3)
    ax2.axhline(0, color="gray", ls="--")
    
    # Limites residuos también inteligentes
    r_low = np.percentile(resid, 1)
    r_high = np.percentile(resid, 99)
    ax2.set_ylim(r_low - 0.005, r_high + 0.005)

    ax2.set_ylabel("Residuos")
    ax2.set_xlabel("Tiempo desde el inicio (Horas)")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plot_file = "transit_fit_v5_cleaned.png"
    plt.savefig(plot_file, dpi=150)
    print(f"✅ Gráfico guardado: {plot_file}")
