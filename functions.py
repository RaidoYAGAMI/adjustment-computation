import math
import numpy as np


def calculate_angle(x1, y1, x2, y2, x3, y3):
    """
    Calculate the angle ABC (in degrees) given the coordinates of points A, B, and C.
    Parameters:
    x1, y1: Coordinates of point A
    x2, y2: Coordinates of point B
    x3, y3: Coordinates of point C
    Returns:
    float: Angle ABC in degrees
    """
    # Vectors BA and BC
    BA_x, BA_y = x1 - x2, y1 - y2
    BC_x, BC_y = x3 - x2, y3 - y2
    # Dot product and magnitudes
    dot_product = BA_x * BC_x + BA_y * BC_y
    magnitude_BA = math.sqrt(BA_x**2 + BA_y**2)
    magnitude_BC = math.sqrt(BC_x**2 + BC_y**2)
    # Cosine of the angle
    cos_angle = dot_product / (magnitude_BA * magnitude_BC)
    # Clamp value to avoid numerical errors (e.g., cos_angle slightly > 1 or < -1)
    cos_angle = max(-1, min(1, cos_angle))
    # Angle in radians and then convert to degrees
    angle_radians = math.acos(cos_angle)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


def decimal_to_dms(decimal_degrees):
    sign = -1 if decimal_degrees < 0 else 1
    decimal_degrees = abs(decimal_degrees)
    # Calculate degrees, minutes, and seconds
    degrees = int(decimal_degrees)
    minutes = int((decimal_degrees - degrees) * 60)
    seconds = (decimal_degrees - degrees - minutes / 60) * 3600
    return [degrees * sign, minutes, seconds]


def dms_to_decimal(dms):
    # Extract degrees, minutes, and seconds
    degrees, minutes, seconds = dms
    # Determine the sign based on degrees
    sign = -1 if degrees < 0 else 1
    decimal_degrees = abs(degrees) + (minutes / 60) + (seconds / 3600)
    return decimal_degrees * sign


def calculate_parameters(xk,yk,xi,yi,xj,yj):
    parameters=np.zeros(6, dtype=float)
    s2_ik=(xi-xk)**2+(yi-yk)**2
    s2_ij=(xi-xj)**2+(yi-yj)**2
    parameters[0]=-(yk-yi)/s2_ik
    parameters[1]=(xk-xi)/s2_ik
    parameters[2]=(yk-yi)/s2_ik-(yj-yi)/s2_ij
    parameters[3]=-(xk-xi)/s2_ik+(xj-xi)/s2_ij
    parameters[4]=(yj-yi)/s2_ij
    parameters[5]=-(xj-xi)/s2_ij

    return parameters


def calculate_side_design_matrix(xi,yi,xj,yj):
    side_parameters=np.zeros(4, dtype=float)
    s_ij = math.sqrt((xi-xj)**2+(yi-yj)**2)
    side_parameters[0] = (xi - xj) / s_ij
    side_parameters[1] = (yi - yj) / s_ij
    side_parameters[2] = (xj - xi) / s_ij
    side_parameters[3] = (yj - yi) / s_ij

    return side_parameters


