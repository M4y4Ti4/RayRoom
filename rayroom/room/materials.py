import numpy as np


class Material:
    """
    Defines the acoustic properties of a surface.
    """

    def __init__(self, name, absorption=0.1, transmission=0.0, scattering=0.0):
        """
        Initialize a Material.

        :param name: Name of the material.
        :type name: str
        :param absorption: Absorption coefficient (alpha). 0 = perfect reflection, 1 = perfect absorption.
                           Can be a single float or array for frequency bands. Defaults to 0.1.
        :type absorption: float or np.array
        :param transmission: Transmission coefficient (tau). 0 = opaque, 1 = fully transparent.
                             Defaults to 0.0.
        :type transmission: float or np.array
        :param scattering: Scattering coefficient (s). 0 = specular, 1 = diffuse. Defaults to 0.0.
        :type scattering: float or np.array
        """
        self.name = name
        self.absorption = np.array(absorption) if isinstance(absorption, (list, tuple)) else np.array([absorption])
        self.transmission = np.array(transmission) if isinstance(
            transmission, (list, tuple)) else np.array([transmission])
        self.scattering = np.array(scattering) if isinstance(scattering, (list, tuple)) else np.array([scattering])

    def __repr__(self):
        return f"Material({self.name}, abs={self.absorption}, trans={self.transmission})"

# Common Materials Library


def get_material(name):
    """
    Retrieve a standard material by name.

    :param name: Name of the material (e.g., "concrete", "glass", "wood").
    :type name: str
    :return: A Material object.
    :rtype: rayroom.materials.Material
    """
    # absorption coeff. taken from ODEON materials database
    # given for frequencies with octave band centre freq. [63, 125, 250, 500, 1000, 2000, 4000]
    materials = {
        "concrete": Material(
        "Concrete",                          # Ref: 100 Rough concrete
        absorption=[0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.07],
        transmission=0.0,
        scattering=0.05
    ),

    "brick": Material(
        "Brick",                             # Ref: 1001 Smooth brickwork with flush pointing
        absorption=[0.02, 0.03, 0.03, 0.04, 0.05, 0.07, 0.07],
        transmission=0.0,
        scattering=0.05
    ),

    "thick_carpet": Material(
        "Thick Carpet",                      # Ref: 7005 Carpet heavy on hairfelt or foam rubber
        absorption=[0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73],
        transmission=0.0,
        scattering=0.2
    ),

    "carpet": Material(
        "Carpet",                            # Ref: 7001 6mm pile carpet bonded to closed-cell foam
        absorption=[0.03, 0.09, 0.25, 0.31, 0.33, 0.44, 0.44],
        transmission=0.0,
        scattering=0.3
    ),

    "glass": Material(
        "Glass",                             # Ref: 10002 Single pane of glass 3mm
        absorption=[0.08, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02],
        transmission=0.1,
        scattering=0.02
    ),

    "heavy_curtain": Material(
        "Heavy Curtain",                     # Ref: 8010 Drapes heavy velour
        absorption=[0.14, 0.35, 0.55, 0.72, 0.70, 0.65, 0.65],
        transmission=0.2,
        scattering=0.5
    ),

    "wood": Material(
        "Wood",                              # Ref: 3004 Wooden floor on joists
        absorption=[0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07],
        transmission=0.01,
        scattering=0.1
    ),

    "plaster": Material(
        "Plaster",                           # Ref: 4000 Lime cement plaster
        absorption=[0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05],
        transmission=0.0,
        scattering=0.05
    ),

    "transparent_wall": Material(
        "TransparentWall",                   # Ref: 10003 Double glazing 2-3mm glass 10mm gap
        absorption=[0.10, 0.07, 0.05, 0.03, 0.02, 0.02, 0.02],
        transmission=0.8,
        scattering=0.0
    ),

    "human": Material(
        "Human",                             # Ref: 11050 Person in suit
        absorption=[0.15, 0.23, 0.56, 0.78, 0.88, 0.89, 0.89],
        transmission=0.0,
        scattering=0.5
    ),

    "asphalt": Material(
        "Asphalt",                           # No direct match — kept as estimate
        absorption=[0.05, 0.10, 0.15, 0.20, 0.20, 0.25, 0.25],
        transmission=0.0,
        scattering=0.1
    ),

    "grass": Material(
        "Grass",                             # No direct match — kept as estimate
        absorption=[0.20, 0.35, 0.55, 0.65, 0.70, 0.75, 0.75],
        transmission=0.0,
        scattering=0.6
    ),

    "soil": Material(
        "Soil",                              # Ref: 9001 Sand 100mm thickness
        absorption=[0.15, 0.35, 0.40, 0.50, 0.55, 0.80, 0.80],
        transmission=0.0,
        scattering=0.7
    ),

    "metal": Material(
        "Metal",                             # Ref: 5000 Steel trapez profile
        absorption=[0.30, 0.25, 0.20, 0.10, 0.10, 0.15, 0.15],
        transmission=0.0,
        scattering=0.1
    ),

    "fabric": Material(
        "Fabric",                            # Ref: 8009 Medium velour draped to half area
        absorption=[0.07, 0.31, 0.49, 0.75, 0.70, 0.60, 0.60],
        transmission=0.05,
        scattering=0.4
    ),

    "leather": Material(
        "Leather",                           # No direct match — kept as estimate
        absorption=[0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.40],
        transmission=0.0,
        scattering=0.2
    ),

    "tempered_glass": Material(
        "Tempered Glass",                    # Ref: 10000 Solid glass blocks
        absorption=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        transmission=0.01,
        scattering=0.05
    ),

    "marble": Material(
        "Marble",                            # Ref: 2001 Marble or glazed tile
        absorption=[0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
        transmission=0.0,
        scattering=0.1
    ),

    "acoustic_foam": Material(
        "Acoustic Foam",                     # Ref: 12002 2.5cm mineral fiber spray-on
        absorption=[0.16, 0.45, 0.70, 0.90, 0.90, 0.85, 0.85],
        transmission=0.0,
        scattering=0.7
    ),

    "drywall": Material(
        "Drywall",                           # Ref: 4042 Plasterboard 13mm 100mm empty cavity
        absorption=[0.08, 0.11, 0.05, 0.03, 0.02, 0.03, 0.03],
        transmission=0.0,
        scattering=0.1
    ),

    "water": Material(
        "Water Surface",                     # Ref: 9000 Water surface in swimming pool
        absorption=[0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
        transmission=0.0,
        scattering=0.1
    ),

    "plywood": Material(
        "Plywood",                           # Ref: 3063 Thin plywood paneling
        absorption=[0.42, 0.21, 0.10, 0.08, 0.06, 0.06, 0.06],
        transmission=0.0,
        scattering=0.15
    ),

    "linoleum": Material(
        "Linoleum",                          # Ref: 6000 Linoleum or vinyl stuck to concrete
        absorption=[0.02, 0.02, 0.03, 0.04, 0.04, 0.05, 0.05],
        transmission=0.0,
        scattering=0.05
    ),

    "ceiling_tile": Material(
        "Ceiling Tile",                      # Ref: 12000 1.27cm mineral fiber spray-on
        absorption=[0.05, 0.15, 0.45, 0.70, 0.80, 0.80, 0.80],
        transmission=0.0,
        scattering=0.6
    ),

    "stucco": Material(
        "Stucco",                            # Ref: 4035 Plaster rough finish on lath
        absorption=[0.14, 0.10, 0.06, 0.05, 0.04, 0.03, 0.03],
        transmission=0.0,
        scattering=0.5
    ),

    "plastic": Material(
        "Plastic",                           # Ref: 102 Smooth concrete painted/glazed (closest hard smooth surface)
        absorption=[0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
        transmission=0.0,
        scattering=0.1
    ),

    "foam_cushion": Material(
        "Foam Cushion",                      # Ref: 11006 Empty chairs upholstered cloth cover
        absorption=[0.44, 0.60, 0.77, 0.89, 0.82, 0.70, 0.70],
        transmission=0.0,
        scattering=0.6
    ),

    "laminate": Material(
        "Laminate",                          # Ref: 3002 Wood parquet in asphalt on concrete
        absorption=[0.04, 0.04, 0.07, 0.06, 0.06, 0.07, 0.07],
        transmission=0.0,
        scattering=0.05
    ),

    "ceramic": Material(
        "Ceramic",                           # Ref: 2001 Marble or glazed tile
        absorption=[0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02],
        transmission=0.0,
        scattering=0.05
    ),
}
    return materials.get(name, Material("Default", 0.1, 0.0, 0.0))
