import sympy as sp
import numpy as np
from math import pi
from scipy.integrate import solve_ivp
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
import time
sp.init_printing()

console = Console()

G, M, r, c = sp.symbols('G M r c', positive=True)
beta = sp.symbols('beta', positive=True)        # GEF coupling; set beta=1
U = G*M/r
kappa = 1 - beta*U/c**2


#######################################################################
# 6.  Integrate equatorial geodesic in κ-metric  (β = γ = 1 case)
#######################################################################
Gm = 6.67408e-11
Mm = 1.98847e30
c2 = 2.99792458e8
a, e = 5.791e10, 0.2056
r0 = a*(1 - e)                    # perihelion radius
v0 = np.sqrt( Gm*Mm * (1+e)/(a*(1-e)) )

# Global variables for progress tracking
progress_bar = None
task_id = None
total_time = 2*pi*88*365*24*3600  # Total integration time
last_update_time = 0

def rhs(t, y):
    global progress_bar, task_id, last_update_time
    
    # y = [r, φ, pr, pφ];   pr = dr/dτ  ,   pφ = r² dφ/dτ
    r, ph, pr, pphi = y
    U = Gm*Mm/r
    # effective potential from κ-metric (weak field)
    g_rr = 1 + 2*U/c2**2
    g_tt = -(1 - 2*U/c2**2)
    # geodesic equations (weak-field approximation)
    dpr = pphi**2/r**3 - Gm*Mm/r**2
    dpphi = 0
    
    # Update progress bar periodically (every 1% of total time)
    current_time = time.time()
    if progress_bar and task_id and (current_time - last_update_time > 0.5):  # Update every 0.5 seconds
        progress_percent = (t / total_time) * 100
        progress_bar.update(task_id, completed=t, description=f"[cyan]Integrating orbit[/cyan] ({progress_percent:.1f}% complete, t={t/365/24/3600:.2f} years)")
        last_update_time = current_time
    
    return [pr, pphi/r**2, dpr, dpphi]

# Setup initial conditions
y0 = [r0, 0, 0, r0*v0]

console.print("[bold green]Starting Mercury perihelion precession calculation...[/bold green]")
console.print(f"[yellow]Integration time span:[/yellow] {total_time/(365*24*3600):.1f} years")
console.print(f"[yellow]Expected result:[/yellow] ~43 arcsec/century (GR prediction)")
console.print()

# Run integration with progress tracking
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console
) as progress:
    progress_bar = progress
    task_id = progress.add_task("[cyan]Integrating orbit[/cyan]", total=total_time)
    
    start_time = time.time()
    orb = solve_ivp(rhs, [0, total_time], y0, rtol=1e-9, atol=1e-9, max_step=10000)
    end_time = time.time()
    
    progress.update(task_id, completed=total_time, description="[green]Integration complete![/green]")

console.print(f"\n[bold green]Integration completed in {end_time - start_time:.2f} seconds[/bold green]")

# Analyze results
phi = orb.y[1]
# detect perihelia via pr≈0
indices = np.where(np.isclose(orb.y[2], 0, atol=1e-5))[0]
peri_angles = phi[indices]
adv = (peri_angles[-1] - peri_angles[0]) - 2*pi*(len(peri_angles)-1)
result = adv*206265*3600/len(peri_angles)

console.print(f"\n[bold cyan]Results:[/bold cyan]")
console.print(f"[yellow]Δϖ direct integration =[/yellow] [bold white]{result:.3f} arcsec/century[/bold white]")
console.print(f"[yellow]Number of perihelia detected:[/yellow] {len(peri_angles)}")
console.print(f"[yellow]Integration success:[/yellow] {'✓' if orb.success else '✗'}")

# Check if result is within target range
if abs(result - 43) <= 0.1:
    console.print(f"[bold green]✓ PASS: Result within target range (43 ± 0.1 arcsec/century)[/bold green]")
else:
    console.print(f"[bold red]✗ FAIL: Result outside target range (43 ± 0.1 arcsec/century)[/bold red]")
#######################################################################