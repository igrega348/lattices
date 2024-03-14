import json
import numpy as np

from e3nn import o3

from . import abaqus
from . import elasticity_func

def test_Mandel_Voigt():
    # Come up with PSD stiffness matrix in Mandel notation
    C = np.random.rand(6,6)
    C = C + C.T # symmetric
    C = C @ C # PSD
    assert np.all(np.linalg.eigvalsh(C) > 0)
    # also test compliance
    S = np.linalg.inv(C)

    # Convert to Voigt notation
    C_voigt = elasticity_func.stiffness_Mandel_to_Voigt(C)
    S_voigt = np.linalg.inv(C_voigt)
    # ensure backward conversion gives the same result
    assert np.allclose(C, elasticity_func.stiffness_Voigt_to_Mandel(C_voigt))
    assert np.allclose(S, elasticity_func.compliance_Voigt_to_Mandel(S_voigt))

    # Ensure the conversion of Voigt and Mandel to 4th order tensors are the same
    C4_v = elasticity_func.stiffness_Voigt_to_4th_order(C_voigt)
    C4_m = elasticity_func.numpy_Mandel_to_cart_4(C)
    assert np.allclose(C4_v, C4_m)
    S4_v = elasticity_func.compliance_Voigt_to_4th_order(S_voigt)
    S4_m = elasticity_func.numpy_Mandel_to_cart_4(S)
    assert np.allclose(S4_v, S4_m)

    # Ensure the conversion back to Voigt and Mandel match
    assert np.allclose(C_voigt, elasticity_func.stiffness_4th_order_to_Voigt(C4_v))
    assert np.allclose(C, elasticity_func.numpy_cart_4_to_Mandel(C4_m))
    assert np.allclose(S_voigt, elasticity_func.compliance_4th_order_to_Voigt(S4_v))
    assert np.allclose(S, elasticity_func.numpy_cart_4_to_Mandel(S4_m))

def test_rotation_rules():
    # Rotations for stress and strain
    sig = np.random.rand(3,3)
    sig = sig + sig.T # symmetric
    Q = o3.rand_matrix().numpy()
    sig_rot = np.einsum('ia,jb,ab->ij', Q, Q, sig)
    sig_Mandel = elasticity_func.tens_2d_to_Mandel_numpy(sig)
    R_mand = elasticity_func.Mandel_rot_matrix_numpy(Q)
    sig_rot_Mandel = R_mand @ sig_Mandel
    assert np.allclose(elasticity_func.tens_2d_to_Mandel_numpy(sig_rot), sig_rot_Mandel)

    # Rotations for stiffness and compliance
    # Come up with PSD stiffness matrix in Mandel notation
    C = np.random.rand(6,6)
    C = C + C.T # symmetric
    C = C @ C # PSD
    assert np.all(np.linalg.eigvalsh(C) > 0)
    C4 = elasticity_func.numpy_Mandel_to_cart_4(C)
    Q = o3.rand_matrix().numpy()
    # check that rotating the stiffness matrix in Mandel notation 
    # gives the same result as rotating the 4th order tensor
    R_mand = elasticity_func.Mandel_rot_matrix_numpy(Q)
    C_rot_Mand = np.einsum('ia,jb,ab->ij', R_mand, R_mand, C)
    C4_rot = elasticity_func.rotate_4th_order(C4, Q)
    assert np.allclose(C_rot_Mand, elasticity_func.numpy_cart_4_to_Mandel(C4_rot))


def test_abaqus_parsing():
    json_text = """
    {
        "Job name": "modified", 
        "Lattice name": "modified", 
        "Base lattice": "E4977", 
        "Date": "2023-12-12", 
        "Relative densities": "0.001, 0.003, 0.01", 
        "Strut radii": "0.0041266, 0.0071475, 0.01305", 
        "Unit cell volume": "1", 
        "Description": "New dataset with B31 and 4 elements per strut", 
        "Imperfection level": "0.0", 
        "Instances": {
            "0": {
                "Relative density": 0.001, 
                "Load-REF1-dof1": {
                    "REF1, RF1": 0.00016054054140113294, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.25005435943603516, 
                    "REF1, U2": -6.959262827876955e-05, 
                    "REF1, U1": 1.0, 
                    "REF2, U1": -0.4997440576553345, 
                    "REF2, U2": 0.00013209973985794932, 
                    "REF2, U3": 0.24974049627780914, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF1-dof2": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0001605456491233781, 
                    "REF1, U3": -0.2500555217266083, 
                    "REF1, U2": 1.0, 
                    "REF1, U1": -6.959484744584188e-05, 
                    "REF2, U1": -0.4997440576553345, 
                    "REF2, U2": -0.2497408539056778, 
                    "REF2, U3": -0.00013329650391824543, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF1-dof3": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 9.264648542739451e-05, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": 1.0, 
                    "REF1, U2": -0.1443001925945282, 
                    "REF1, U1": -0.144304096698761, 
                    "REF2, U1": -0.059578679502010345, 
                    "REF2, U2": -0.21364915370941162, 
                    "REF2, U3": 0.21364842355251312, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof1": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.042791012674570084, 
                    "REF1, U2": -0.20712846517562866, 
                    "REF1, U1": -0.20713505148887634, 
                    "REF2, U1": 0.5, 
                    "REF2, U2": 0.1980707049369812, 
                    "REF2, U3": -0.19807113707065582, 
                    "REF2, RF1": 6.654120807070285e-05, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof2": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.16715699434280396, 
                    "REF1, U2": -0.11275697499513626, 
                    "REF1, U1": 5.964439333183691e-05, 
                    "REF2, U1": 0.21576546132564545, 
                    "REF2, U2": 0.5, 
                    "REF2, U3": -0.21782620251178741, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 7.248570182127878e-05, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof3": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": 0.16715654730796814, 
                    "REF1, U2": -6.0182872402947396e-05, 
                    "REF1, U1": 0.11276048421859741, 
                    "REF2, U1": -0.21576610207557678, 
                    "REF2, U2": -0.21782636642456055, 
                    "REF2, U3": 0.5, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 7.24857600289397e-05
                }
            }, 
            "1": {
                "Relative density": 0.003, 
                "Load-REF1-dof1": {
                    "REF1, RF1": 0.0004818919114768505, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.2501613199710846, 
                    "REF1, U2": -0.0002080604899674654, 
                    "REF1, U1": 1.0, 
                    "REF2, U1": -0.49923449754714966, 
                    "REF2, U2": 0.00039399389061145484, 
                    "REF2, U3": 0.24922484159469604, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF1-dof2": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.00048193789552897215, 
                    "REF1, U3": -0.2501625120639801, 
                    "REF1, U2": 1.0, 
                    "REF1, U1": -0.0002080803387798369, 
                    "REF2, U1": -0.49923455715179443, 
                    "REF2, U2": -0.24922530353069305, 
                    "REF2, U3": -0.0003951542603317648, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF1-dof3": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0002781372459139675, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": 1.0, 
                    "REF1, U2": -0.14437443017959595, 
                    "REF1, U1": -0.14438752830028534, 
                    "REF2, U1": -0.05923108011484146, 
                    "REF2, U2": -0.21325600147247314, 
                    "REF2, U3": 0.21325483918190002, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof1": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.0425899513065815, 
                    "REF1, U2": -0.20717158913612366, 
                    "REF1, U1": -0.2071913480758667, 
                    "REF2, U1": 0.5, 
                    "REF2, U2": 0.1977667361497879, 
                    "REF2, U3": -0.19777041673660278, 
                    "REF2, RF1": 0.00019999385403934866, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof2": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.16699500381946564, 
                    "REF1, U2": -0.11263217031955719, 
                    "REF1, U1": 0.00017807428957894444, 
                    "REF2, U1": 0.21537631750106812, 
                    "REF2, U2": 0.5, 
                    "REF2, U3": -0.21758082509040833, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0002178017603000626, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof3": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": 0.1669931560754776, 
                    "REF1, U2": -0.00017858071078080684, 
                    "REF1, U1": 0.1126420721411705, 
                    "REF2, U1": -0.21537911891937256, 
                    "REF2, U2": -0.21757960319519043, 
                    "REF2, U3": 0.5, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.00021780053793918341
                }
            }, 
            "2": {
                "Relative density": 0.01, 
                "Load-REF1-dof1": {
                    "REF1, RF1": 0.0016094319289550185, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.2505227327346802, 
                    "REF1, U2": -0.000685299513861537, 
                    "REF1, U1": 1.0, 
                    "REF2, U1": -0.4974755346775055, 
                    "REF2, U2": 0.0012912880629301071, 
                    "REF2, U3": 0.24744495749473572, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF1-dof2": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0016099425265565515, 
                    "REF1, U3": -0.2505238354206085, 
                    "REF1, U2": 1.0, 
                    "REF1, U1": -0.0006855169194750488, 
                    "REF2, U1": -0.49747559428215027, 
                    "REF2, U2": -0.24744576215744019, 
                    "REF2, U3": -0.0012923481408506632, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF1-dof3": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0009294146439060569, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": 1.0, 
                    "REF1, U2": -0.14462660253047943, 
                    "REF1, U1": -0.14467184245586395, 
                    "REF2, U1": -0.058035850524902344, 
                    "REF2, U2": -0.2118988335132599, 
                    "REF2, U3": 0.21189630031585693, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof1": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.041895490139722824, 
                    "REF1, U2": -0.20732030272483826, 
                    "REF1, U1": -0.2073860466480255, 
                    "REF2, U1": 0.5, 
                    "REF2, U2": 0.19671383500099182, 
                    "REF2, U3": -0.19672878086566925, 
                    "REF2, RF1": 0.0006709349690936506, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof2": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": -0.16643281280994415, 
                    "REF1, U2": -0.11219914257526398, 
                    "REF1, U1": 0.0005856934585608542, 
                    "REF2, U1": 0.21402984857559204, 
                    "REF2, U2": 0.5, 
                    "REF2, U3": -0.2167327105998993, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0007299950229935348, 
                    "REF2, RF3": 0.0
                }, 
                "Load-REF2-dof3": {
                    "REF1, RF1": 0.0, 
                    "REF1, RF3": 0.0, 
                    "REF1, RF2": 0.0, 
                    "REF1, U3": 0.16642619669437408, 
                    "REF1, U2": -0.000585972098633647, 
                    "REF1, U1": 0.11223124712705612, 
                    "REF2, U1": -0.21404016017913818, 
                    "REF2, U2": -0.21672669053077698, 
                    "REF2, U3": 0.5, 
                    "REF2, RF1": 0.0, 
                    "REF2, RF2": 0.0, 
                    "REF2, RF3": 0.000729974708519876
                }
            }
        }
    }
    """

    dd = json.loads(json_text)
    S_m = abaqus.calculate_compliance_Mandel(dd['Instances']['0'], float(dd['Unit cell volume']))
    S_v = abaqus.calculate_compliance_Voigt(dd['Instances']['0'], float(dd['Unit cell volume']))
    S_4 = elasticity_func.compliance_Voigt_to_4th_order(S_v)
    S_4m = elasticity_func.numpy_Mandel_to_cart_4(S_m)
    assert np.allclose(S_4, S_4m)