# Guía de identidad visual — Adaptación a app de escritorio en Python

Este documento describe la identidad, paleta, tipografía y componentes del proyecto base para que puedan ser **replicados en una app de escritorio en Python**, sin importar la librería gráfica que se use (PySide6/PyQt, Tkinter/CustomTkinter, Kivy/KivyMD, wxPython, Toga, Dear PyGui, etc.).

La guía está escrita en términos **agnósticos al framework**: define tokens, reglas de uso y patrones de layout. Al final hay ejemplos por framework para acelerar la implementación.

---

## 1. Identidad de marca

- **Color de marca:** gris humo/nube, hue `210` en HSL (≈ `#8B9BAA` en HEX). *Antes era violeta `#9B7BD8`; cambiado en 2026-05 para una estética más sobria y neutra.*
- **Estilo general:** minimalismo neutro + un único color de acento gris azulado. Mucho gris, mucho blanco/negro neutro y gris humo para resaltar acciones.
- **Esquinas redondeadas:** estilo iOS — radio **10 px** (botones/inputs), **14 px** (cards/paneles), **18 px** (GlassCard/dialogs heroicos). *Valores anteriores: 6 px / 8 px / 12 px.*
- **Sombras:** suaves, nunca duras. Las cards heroicas (login, modales) sí pueden llevar sombra grande.
- **Modo claro y oscuro:** **obligatorio**. Toda la UI tiene que funcionar en ambos.
- **Tipografía:** `"Segoe UI Symbol"` primero (garantiza glifos monocromo), luego `"Segoe UI"`, `"Inter"`, `"Helvetica Neue"`, `"Arial Unicode MS"`, `sans-serif`. Jerarquía marcada con tamaños y pesos (no con colores).
- **Iconografía:** BMP Unicode (plano 0, U+0000–U+FFFF) con variante selector U+FE0E para forzar renderizado monocromo. Definidos en `prospective/ui/icons.py` (clase `_Icons`). Fuente: `"Segoe UI Symbol"`. *Antes: emoji de plano alto (U+1F000+) que el OS renderizaba en color.*
- **Logo:** dos versiones (clara y oscura). Se intercambian según el tema activo.
- **Idioma de la UI:** español neutro.

---

## 2. Paleta de colores (tokens)

> No uses HEX hardcodeado en cada pantalla. Define estos tokens una sola vez (en un módulo `tokens.py` o equivalente) y referencia siempre por nombre. Eso permite cambiar tema y mantener consistencia.

### Tema claro

| Token                    | HEX        | Para qué se usa |
|--------------------------|------------|-----------------|
| `background`             | `#FFFFFF`  | fondo de la ventana principal |
| `foreground`             | `#0D0D0D`  | texto principal |
| `card`                   | `#F7F7F7`  | superficies elevadas (paneles, cards) |
| `card_foreground`        | `#0D0D0D`  | texto sobre card |
| `popover`                | `#FFFFFF`  | menús, tooltips, dropdowns |
| `popover_foreground`     | `#0D0D0D`  | texto en popover |
| `primary`                | `#8B9BAA`  | color de marca: botones primarios, foco, links |
| `primary_foreground`     | `#1C1C1C`  | texto sobre primary |
| `secondary`              | `#F2F2F2`  | botones/superficies neutras |
| `secondary_foreground`   | `#1C1C1C`  | texto sobre secondary |
| `muted`                  | `#F2F2F2`  | fondo discreto (cabeceras de tabla, badges suaves) |
| `muted_foreground`       | `#6B6B6B`  | texto secundario / placeholders |
| `accent`                 | `#DDE5EC`  | hover sutil, badges informativos |
| `accent_foreground`      | `#2E4A5F`  | texto/icono sobre accent |
| `destructive`            | `#D7263D`  | acciones destructivas y errores |
| `border`                 | `#E5E5E5`  | bordes de inputs, cards y separadores |
| `input`                  | `#E5E5E5`  | borde de inputs (alias de `border`) |
| `ring`                   | `#8B9BAA`  | anillo de foco |
| `chart_1`                | `#2A4C66`  | gráficas (más oscuro) |
| `chart_2`                | `#3E6480`  | gráficas |
| `chart_3`                | `#8B9BAA`  | gráficas (color de marca) |
| `chart_4`                | `#A8B8C6`  | gráficas (más claro) |
| `chart_5`                | `#305070`  | gráficas (saturado) |

### Tema oscuro

| Token                    | HEX        |
|--------------------------|------------|
| `background`             | `#1F1F1F`  |
| `foreground`             | `#EBEBEB`  |
| `card`                   | `#2A2A2A`  |
| `card_foreground`        | `#EBEBEB`  |
| `popover`                | `#2A2A2A`  |
| `popover_foreground`     | `#EBEBEB`  |
| `primary`                | `#A8B8C6`  |
| `primary_foreground`     | `#1C1C1C`  |
| `secondary`              | `#363636`  |
| `secondary_foreground`   | `#EBEBEB`  |
| `muted`                  | `#363636`  |
| `muted_foreground`       | `#9B9B9B`  |
| `accent`                 | `#363636`  |
| `accent_foreground`      | `#EBEBEB`  |
| `destructive`            | `#E85A6A`  |
| `border`                 | `#363636`  |
| `input`                  | `#363636`  |
| `ring`                   | `#8B9BAA`  |
| `chart_1`                | `#C0D2E0`  |
| `chart_2`                | `#A4BCD0`  |
| `chart_3`                | `#8B9BAA`  |
| `chart_4`                | `#4E6A84`  |
| `chart_5`                | `#305070`  |

### Color de selección de texto

- Fondo: `primary`
- Texto: `primary_foreground`

### Reglas de oro

1. **Nunca** uses `#FFFFFF` o `#000000` directos en widgets. Usa `background`/`foreground`.
2. Para hover de botones primarios, oscurece `primary` un 10 % (≈ `#6A8399` en light → `#4E6678` en dark).
3. Para hover de botones outline/ghost, fondo = `accent`.
4. Un texto secundario siempre va en `muted_foreground`, nunca en gris arbitrario.
5. Errores y campos inválidos: borde y halo en `destructive`.

---

## 3. Tipografía

- **Familia:** preferencia `Inter`, fallback `Segoe UI`, luego `system-ui`, `Avenir`, `Helvetica`, `Arial`, `sans-serif`.
- **Suavizado:** activar antialias / subpixel rendering si el framework lo permite.
- **Letter‑spacing:** títulos con tracking apretado (`-0.02em`); cuerpo y UI sin tracking.

### Escala recomendada

| Rol               | Tamaño | Peso        | Color                |
|-------------------|--------|-------------|----------------------|
| Hero / Login      | 22 px  | 800 (extra) | `foreground`         |
| Título de página  | 18 px  | 800 (extra) | `foreground`         |
| Título de sección | 16 px  | 600 (semi)  | `foreground`         |
| Cuerpo            | 14 px  | 400         | `foreground`         |
| Descripción       | 13 px  | 400         | `muted_foreground`   |
| Etiqueta pequeña  | 11 px  | 600 (semi), `UPPERCASE` | `muted_foreground` |

---

## 4. Geometría y espaciado

### Radios

> Escala actualizada en 2026-05 a estilo iOS. Valores anteriores indicados entre paréntesis.

| Token         | px    | Uso                                             |
|---------------|-------|-------------------------------------------------|
| `radius_sm`   | 6     | micro badges, items de menú, checkboxes *(antes 4)* |
| `radius_md`   | 10    | botones, inputs, combobox, tooltip *(antes 6)*   |
| `radius_lg`   | 14    | cards, paneles, listas, tablas, tabs *(antes 8)* |
| `radius_xl`   | 18    | GlassCard, dialogs heroicos, sheets *(antes 12)* |
| `radius_full` | 9999  | badges píldora / avatares                       |

### Espaciado

| Token   | px |
|---------|----|
| `xs`    | 4  |
| `sm`    | 8  |
| `md`    | 12 |
| `lg`    | 16 |
| `xl`    | 24 |
| `2xl`   | 32 |

- Padding interior estándar de cards: 24 px (`xl`).
- Separación vertical entre bloques de una página: 24 px.
- Padding lateral del contenedor de página: 16 px.
- Ancho máximo de contenido: 1280 px centrado.

### Sombras

| Token       | Definición CSS                          | Uso                       |
|-------------|------------------------------------------|---------------------------|
| `shadow_xs` | `0 1px 2px rgba(0,0,0,0.05)`             | inputs y botones outline  |
| `shadow_sm` | `0 1px 3px rgba(0,0,0,0.08)`             | cards estándar            |
| `shadow_md` | `0 4px 12px rgba(0,0,0,0.10)`            | popovers, tooltips        |
| `shadow_lg` (light) | `0 20px 60px rgba(2,6,23,0.35)` | login card, modales       |
| `shadow_lg` (dark)  | `0 20px 70px rgba(0,0,0,0.55)` | login card en dark        |

### Duraciones de animación

| Token        | ms  |
|--------------|-----|
| `fast`       | 150 |
| `base`       | 300 |
| `slow`       | 500 |
| `slowest`    | 700 |

---

## 5. Componentes (especificación funcional, sin código)

### 5.1. Botón

Variantes:

- **default (primario):** fondo `primary`, texto `primary_foreground`, hover oscurecido 10 %.
- **outline:** fondo transparente, borde 1 px `border`, texto `foreground`, hover fondo `accent`.
- **secondary:** fondo `secondary`, texto `secondary_foreground`, hover ligeramente oscurecido.
- **ghost:** sin fondo ni borde, hover fondo `accent`.
- **destructive:** fondo `destructive`, texto blanco.
- **link:** sin fondo, texto `primary`, subrayado en hover.

Tamaños:

| Tamaño  | Alto | Padding horizontal | Texto |
|---------|------|--------------------|-------|
| `sm`    | 32 px | 12 px              | 13 px |
| `md` (default) | 36 px | 16 px       | 14 px |
| `lg`    | 40 px | 24 px              | 14 px |
| `icon`  | 36 px (cuadrado) | —        | —     |

- Esquinas: `radius_md` (10 px).
- Estado deshabilitado: opacidad 50 %, cursor `not-allowed`.
- Foco: borde `ring` + halo de 3 px con `ring` al 50 % de alpha.

### 5.2. Input / TextField

- Alto: 36 px.
- Borde: 1 px `input`, esquinas `radius_md` (10 px).
- Fondo: transparente (en dark, fondo `input` con 30 % alpha).
- Texto: `foreground`. Placeholder: `muted_foreground`.
- Foco: borde cambia a `ring` + halo 3 px alpha 50 %.
- Estado inválido: borde `destructive` + halo `destructive` 20 % alpha.

### 5.3. Card

- Fondo: `card`.
- Borde: 1 px `border`.
- Radio: `radius_lg` (14 px) para cards estándar; `radius_xl` (18 px) para GlassCard/dialogs heroicos.
- Padding interior: 24 px.
- Sombra: `shadow_sm` por defecto, `shadow_lg` para heroicas.
- Estructura típica: `header` (título + descripción + acciones) ⇒ separador opcional ⇒ `content` ⇒ `footer` opcional.

### 5.4. Badge

- Forma píldora (`radius_full`).
- Padding: 8 px horizontal, 2 px vertical.
- Texto 11 px peso 500.
- Variantes: `default` (primary), `secondary`, `destructive`, `outline`.

### 5.5. Separador

- 1 px de grosor, color `border`. Horizontal o vertical según contexto.

### 5.6. Tooltip / Popover

- Fondo `popover`, texto `popover_foreground`, borde 1 px `border`.
- Radio `radius_md`, sombra `shadow_md`.
- Padding 8 px horizontal, 6 px vertical.

### 5.7. Tabla

- Cabecera: fondo `muted`, texto `muted_foreground`, peso 600, `UPPERCASE`, 11 px.
- Filas: fondo `card`, separador 1 px `border` entre filas.
- Selección de fila: fondo `accent`, texto `accent_foreground`.

### 5.8. Topbar (cabecera)

- Alto ~56 px.
- Fondo translúcido: `background` con 95 % de alpha + blur de 8 px (si el framework lo permite). En dark, alpha 90 %.
- Borde inferior 1 px `border`.
- Layout: logo a la izquierda, identidad/usuario al centro o derecha, botones de menú/tema/salir en el extremo derecho.

### 5.9. Sheet / Drawer (panel lateral)

- Ancho 420 px (en pantallas anchas) o 100 % en pantallas pequeñas.
- Fondo `background` al 95 % alpha + blur.
- Borde izquierdo 1 px `border`.
- Items de navegación como botones `outline`; el activo cambia a variante `default` (primary).

### 5.10. Toggle de tema

- Botón `outline` cuadrado tamaño `icon`.
- Icono: sol cuando el tema actual es oscuro, luna cuando es claro.
- Persistir la elección entre sesiones.

---

## 6. Layout de páginas

### Page shell estándar

1. Contenedor centrado, ancho máximo 1280 px, padding lateral 16 px, padding vertical 24 px.
2. Primera card en la parte superior con:
   - Título 18 px peso 800.
   - Descripción 13 px en `muted_foreground`.
   - Botones de acción a la derecha (en pantallas anchas) o debajo (en estrechas).
3. Debajo, los bloques de contenido con `space-y` de 24 px entre ellos.

### Página de login / autenticación

- Ventana centrada vertical y horizontalmente.
- Card 420 px de ancho, padding 32 px, radio `radius_xl`, sombra `shadow_lg`.
- Cabecera: logo a la izquierda + toggle de tema a la derecha.
- Título hero 22 px peso 800.
- Subtítulo 13 px `muted_foreground`.
- Formulario apilado verticalmente con `gap` de 16 px.
- Botón de envío ancho completo, primario.
- Mensaje de error: 11 px centrado, color `destructive`.
- Mensaje de éxito/info: 11 px centrado, verde (`#10B981`).

---

## 7. Animaciones y micro‑interacciones

| Interacción                    | Especificación                                                                 |
|--------------------------------|--------------------------------------------------------------------------------|
| Aparición de panel/card        | Fade-in + desplazamiento de 4 px desde arriba, duración `slow` (500 ms), easing out. |
| Hover en botón de menú lateral | Desplazamiento de +4 px en X, duración `base` (300 ms).                        |
| Hover en botón de icono        | Escala 1.05 + sombra suave, duración `base`.                                   |
| Error de formulario            | Shake horizontal ±8 px durante 420 ms.                                         |
| Éxito de formulario            | Pulso de escala a 1.01 con sombra verde durante 500 ms.                        |
| Cambio de tema                 | Crossfade entre paletas, 200 ms.                                               |
| Foco de input                  | Crecimiento del halo del ring de 0 a 3 px, duración `fast` (150 ms).            |
| Apertura de drawer             | Slide desde el borde derecho, duración `base`, easing out.                     |

---

## 8. Sistema de tema (cómo gestionarlo)

Recomendación independiente del framework:

1. Crear un módulo `theme.py` con:
   - Diccionarios `LIGHT` y `DARK` con todos los tokens.
   - Función `get_theme()` que devuelve el dict activo.
   - Función `set_theme(name)` que persiste la elección y dispara un evento.
   - Función `initialize_theme()` que aplica el tema guardado al arrancar.
2. Persistir la preferencia entre sesiones. Opciones según stack:
   - Qt: `QSettings`.
   - General: archivo JSON en la carpeta de configuración del usuario (usar `appdirs` o `platformdirs`).
3. Detectar el tema del sistema operativo para usarlo como valor por defecto la primera vez.
4. Emitir una señal/evento `theme_changed` para que los widgets se redibujen.

---

## 9. Iconos y assets

- **Iconos:** BMP Unicode (plano 0, rango U+0000–U+FFFF) con *Variation Selector-15* (`U+FE0E`) para forzar renderizado monocromo. Definidos centralizadamente en `prospective/ui/icons.py` (clase `_Icons`, alias `I`). La fuente `"Segoe UI Symbol"` debe ir primera en el stack para garantizar glifos grises en lugar de emoji a color. *Los emoji de plano alto (U+1F000+) no pueden forzarse monocromo y deben evitarse.*

  Iconos principales disponibles:

  | Constante        | Símbolo | Uso                          |
  |------------------|---------|------------------------------|
  | `I.SEED`         | ⊙       | Punto semilla MPR            |
  | `I.GROWTH`       | ↑       | Segmentar / crecer           |
  | `I.MEASURE`      | ↔       | Nueva medición               |
  | `I.EYE`          | ◉       | Mostrar/visible              |
  | `I.EYE_HIDDEN`   | ◌       | Ocultar                      |
  | `I.ANGLE_MEAS`   | ∠       | Medir ángulos / secciones    |
  | `I.BRAIN`        | ✺       | Perforantes / neural         |
  | `I.CLIPS`        | ✂︎      | Clips / corte                |
  | `I.FOLDER`       | ⊡       | Abrir / cargar               |
  | `I.SAVE`         | ⊞       | Guardar / anotación          |
  | `I.REFRESH`      | ↻       | Resetear / recargar          |
  | `I.THEME_LIGHT`  | ◐       | Toggle a tema claro          |
  | `I.THEME_DARK`   | ◑       | Toggle a tema oscuro         |

- **Logo:** se necesitan dos versiones (clara para fondo oscuro, oscura para fondo claro), preferiblemente PNG con transparencia. Mostrar la que corresponda según el tema activo.
- **Fuente:** stack completo: `"Segoe UI Symbol"`, `"Segoe UI"`, `"Inter"`, `"Helvetica Neue"`, `"Arial Unicode MS"`, `sans-serif`. `"Segoe UI Symbol"` debe ir primero para garantizar glifos monocromo. `"Inter"` puede embeberse opcionalmente en `assets/fonts/`.

---

## 10. Ejemplos de implementación por framework

> Estos ejemplos son orientativos; lo importante son los tokens y reglas de las secciones anteriores.

### 10.1. PyQt5 / PySide6 (QSS)

```python
from prospective.ui.icons import I  # BMP Unicode icons — monochrome guaranteed

LIGHT = {
    "background": "#FFFFFF", "foreground": "#0D0D0D",
    "card": "#F7F7F7", "border": "#E5E5E5",
    "primary": "#8B9BAA", "primary_foreground": "#1C1C1C",
    "muted_foreground": "#6B6B6B", "accent": "#DDE5EC",
    "accent_foreground": "#2E4A5F",
    "destructive": "#D7263D", "ring": "#8B9BAA",
}

QSS = f"""
* {{
    font-family: "Segoe UI Symbol","Segoe UI","Inter","Helvetica Neue","Arial Unicode MS",sans-serif;
    font-size: 14px;
    color: {LIGHT["foreground"]};
}}
QMainWindow, QDialog, QWidget {{ background-color: {LIGHT["background"]}; }}
QFrame#Card {{
    background-color: {LIGHT["card"]};
    border: 1px solid {LIGHT["border"]};
    border-radius: 14px;          /* radius_lg */
}}
QPushButton {{
    background-color: {LIGHT["primary"]}; color: {LIGHT["primary_foreground"]};
    border: none; border-radius: 10px;   /* radius_md */
    padding: 8px 16px; font-weight: 500;
}}
QPushButton:hover {{ background-color: #6A8399; }}
QPushButton[variant="outline"] {{
    background: transparent; color: {LIGHT["foreground"]};
    border: 1px solid {LIGHT["border"]};
}}
QPushButton[variant="outline"]:hover {{ background-color: {LIGHT["accent"]}; }}
QLineEdit {{
    background: transparent; border: 1px solid {LIGHT["border"]};
    border-radius: 10px;          /* radius_md */
    padding: 6px 10px; color: {LIGHT["foreground"]};
    selection-background-color: {LIGHT["primary"]};
    selection-color: {LIGHT["primary_foreground"]};
}}
QLineEdit:focus {{ border: 1px solid {LIGHT["ring"]}; }}
QToolTip {{
    background-color: #FFFFFF; color: {LIGHT["foreground"]};
    border: 1px solid {LIGHT["border"]};
    border-radius: 10px;          /* radius_md */
    padding: 6px 8px;
}}
"""

app.setStyleSheet(QSS)
```

Para el halo de foco, usar `QGraphicsDropShadowEffect` con color `ring` y blur ~12 px cuando el widget recibe foco.

### 10.2. CustomTkinter

```python
import customtkinter as ctk

ctk.set_appearance_mode("system")  # o "light" / "dark"

btn = ctk.CTkButton(
    parent, text="Continuar",
    fg_color="#8B9BAA", hover_color="#6A8399",
    text_color="#1C1C1C", corner_radius=10, height=36,  # radius_md
)

card = ctk.CTkFrame(
    parent,
    fg_color=("#F7F7F7", "#2A2A2A"),     # (light, dark)
    border_color=("#E5E5E5", "#363636"),
    border_width=1, corner_radius=14,    # radius_lg
)

entry = ctk.CTkEntry(
    parent, fg_color="transparent",
    border_color=("#E5E5E5", "#363636"),
    text_color=("#0D0D0D", "#EBEBEB"),
    placeholder_text_color=("#6B6B6B", "#9B9B9B"),
    corner_radius=10, height=36,         # radius_md
)
```

CustomTkinter acepta tuplas `(light, dark)` en casi todas las propiedades de color.

### 10.3. Tkinter puro (ttk)

```python
from tkinter import ttk
import tkinter as tk

root = tk.Tk()
style = ttk.Style(root)
style.theme_use("clam")

LIGHT = {
    "bg": "#FFFFFF", "fg": "#0D0D0D",
    "card": "#F7F7F7", "border": "#E5E5E5",
    "primary": "#8B9BAA", "primary_fg": "#1C1C1C",
}

root.configure(bg=LIGHT["bg"])
style.configure("Primary.TButton",
    background=LIGHT["primary"], foreground=LIGHT["primary_fg"],
    borderwidth=0, padding=(16, 8), font=("Segoe UI Symbol", 10, "normal"))
style.map("Primary.TButton", background=[("active", "#6A8399")])
style.configure("Card.TFrame", background=LIGHT["card"], relief="flat")
style.configure("TEntry", fieldbackground=LIGHT["bg"], foreground=LIGHT["fg"],
    bordercolor=LIGHT["border"], lightcolor=LIGHT["border"], darkcolor=LIGHT["border"])
# Nota: ttk no soporta border-radius nativo; usar tk.Canvas para esquinas redondeadas.
```

### 10.4. KivyMD

```python
from kivymd.app import MDApp

class App(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "BlueGrey"  # antes DeepPurple
        self.theme_cls.primary_hue = "400"
        self.theme_cls.accent_palette = "BlueGrey"
```

Para fidelidad exacta, sobreescribir colores en KV usando los HEX de la sección 2. Conversión:

```python
def hex_to_rgba(hex_str, a=1.0):
    h = hex_str.lstrip("#")
    return (int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255, a)
```

### 10.5. wxPython / Toga / Dear PyGui

Aplicar la misma idea: definir un diccionario con los tokens y aplicarlos a través de la API de estilo del framework. La paleta y reglas son las mismas.

---

## 11. Checklist de adaptación

- [ ] Crear módulo de tokens (`theme.py`) con paletas `LIGHT` y `DARK` completas.
- [ ] Implementar gestor de tema con persistencia y evento de cambio.
- [ ] Embeber o cargar fuente `Inter` (opcional pero recomendado).
- [ ] Conseguir las dos versiones del logo (clara y oscura) del nuevo proyecto.
- [ ] Crear primitivos reutilizables: `Button(variant=...)`, `Card`, `Input`, `Badge`, `Separator`, `Topbar`, `PageShell`, `Sheet`, `ThemeToggle`. Todos deben consumir solo tokens.
- [ ] Descargar iconos Lucide a `assets/icons/` y aplicar tinte programático.
- [ ] Verificar contraste AA en ambos temas (texto sobre `card`, sobre `primary`, sobre `destructive`).
- [ ] Implementar las micro‑interacciones de la sección 7 (al menos: hover, foco, error shake, fade-in de panel).
- [ ] Probar la app en light y dark, y que el cambio de tema sea instantáneo y completo (sin widgets sueltos con colores antiguos).

---

## 12. Resumen ejecutivo

> **Color de marca:** gris humo `#8B9BAA` *(antes violeta `#9B7BD8`)*. **Neutros:** blanco/negro suaves más grises `#F7F7F7` y `#E5E5E5`. **Radios (estilo iOS):** 10 px botones/inputs, 14 px cards/paneles, 18 px GlassCard *(antes 6 / 8 / 12 px)*. **Tipografía:** `"Segoe UI Symbol"` primero (iconos monocromo), luego Inter/Segoe UI, títulos en 800. **Sombras:** suaves. **Dark mode obligatorio.** **Iconos:** BMP Unicode monocromo (módulo `icons.py`) *(antes Lucide SVG / emoji color)*. **Animaciones:** sutiles, 150–500 ms. **Patrón base:** topbar translúcida + page shell con card de cabecera + bloques con separación de 24 px.

Cualquier framework Python sirve mientras se respeten estos tokens y reglas: el resultado se sentirá parte de la misma familia visual que el proyecto base.

---

## 13. Historial de cambios de identidad visual

| Fecha     | Cambio | Archivos principales afectados |
|-----------|--------|-------------------------------|
| 2026-05   | **Paleta:** violeta `#9B7BD8` → gris humo `#8B9BAA` y toda su escala derivada | `themes.py`, 25 archivos de UI |
| 2026-05   | **Iconos:** emoji color (U+1F000+) → BMP Unicode monocromo con `U+FE0E`; fuente `"Segoe UI Symbol"` primera en stack | `icons.py`, `themes.py`, todos los paneles/ventanas |
| 2026-05   | **Border-radius:** escala iOS — botones 6→10 px, cards 8→14 px, dialogs 12→18 px | `themes.py`, 20 archivos de UI |
