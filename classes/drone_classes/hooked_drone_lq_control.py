import numpy as np
import scipy as si
import sympy as sp
import control
from classes.controller_base import ControllerBase


class LqrLoadControl(ControllerBase):
    def __init__(self, mass, inertia, gravity):
        super().__init__(mass, inertia, gravity)
        self.gravity = np.abs(self.gravity[2])
        self.mass = self.mass[0]
        self.payload_mass = 0.05
        self.L = 0.4

        # Weight matrices for discrete time LQR
        self.Q_d = np.diag(np.hstack((50 * np.ones(3), 1 * np.ones(3), 1 * np.ones(3), np.ones(3),
                                      1 * np.ones(2), 1 * np.ones(2))))
        self.R_d = 1 * np.eye(4)

        self.dt = 0.01

        self.K_lti = np.zeros((self.R_d.shape[0], self.Q_d.shape[0]))
        self.compute_lti_lqr()
        self.K = None
        self.controller_step = 0

    def funcA(self, _Dummy_482, _Dummy_481, _Dummy_480, _Dummy_476, _Dummy_475, _Dummy_474, _Dummy_488, _Dummy_483,
              _Dummy_487, _Dummy_491, _Dummy_490, _Dummy_489, _Dummy_493, _Dummy_492, _Dummy_478, _Dummy_477,
              _Dummy_494, _Dummy_486, _Dummy_485, _Dummy_484, m, mL, g, L, Jx, Jy, Jz, _Dummy_495, _Dummy_479):
        return np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, -(-_Dummy_494 * (
                         np.sin(_Dummy_483) * np.sin(_Dummy_487) * np.sin(_Dummy_488) + np.cos(_Dummy_487) * np.cos(_Dummy_488)) * np.sin(
                 _Dummy_493) - _Dummy_494 * (
                                     -np.sin(_Dummy_483) * np.sin(_Dummy_488) * np.cos(_Dummy_487) + np.sin(_Dummy_487) * np.cos(
                                 _Dummy_488)) * np.sin(_Dummy_492) * np.cos(_Dummy_493) + _Dummy_494 * np.sin(_Dummy_488) * np.cos(
                 _Dummy_483) * np.cos(_Dummy_492) * np.cos(_Dummy_493)) * np.sin(_Dummy_492) * np.cos(_Dummy_493) / (m + mL), (
                          -_Dummy_494 * (np.sin(_Dummy_483) * np.sin(_Dummy_487) * np.sin(_Dummy_488) + np.cos(_Dummy_487) * np.cos(
                      _Dummy_488)) * np.sin(_Dummy_493) - _Dummy_494 * (
                                      -np.sin(_Dummy_483) * np.sin(_Dummy_488) * np.cos(_Dummy_487) + np.sin(_Dummy_487) * np.cos(
                                  _Dummy_488)) * np.sin(_Dummy_492) * np.cos(_Dummy_493) + _Dummy_494 * np.sin(_Dummy_488) * np.cos(
                      _Dummy_483) * np.cos(_Dummy_492) * np.cos(_Dummy_493)) * np.sin(_Dummy_493) / (m + mL), -(-_Dummy_494 * (
                         np.sin(_Dummy_483) * np.sin(_Dummy_487) * np.sin(_Dummy_488) + np.cos(_Dummy_487) * np.cos(_Dummy_488)) * np.sin(
                 _Dummy_493) - _Dummy_494 * (-np.sin(_Dummy_483) * np.sin(_Dummy_488) * np.cos(_Dummy_487) + np.sin(
                 _Dummy_487) * np.cos(_Dummy_488)) * np.sin(_Dummy_492) * np.cos(_Dummy_493) + _Dummy_494 * np.sin(
                 _Dummy_488) * np.cos(_Dummy_483) * np.cos(_Dummy_492) * np.cos(_Dummy_493)) * np.cos(_Dummy_492) * np.cos(
                 _Dummy_493) / (m + mL),
              (-_Dummy_489 * np.sin(_Dummy_488) + _Dummy_490 * np.cos(_Dummy_488)) * np.tan(_Dummy_483),
              -_Dummy_489 * np.cos(_Dummy_488) - _Dummy_490 * np.sin(_Dummy_488),
              (-_Dummy_489 * np.sin(_Dummy_488) + _Dummy_490 * np.cos(_Dummy_488)) / np.cos(_Dummy_483), 0, 0, 0, 0, 0,
              _Dummy_494 * (
                          np.sin(_Dummy_483) * np.sin(_Dummy_487) * np.sin(_Dummy_488) * np.cos(_Dummy_493) + np.sin(_Dummy_483) * np.sin(
                      _Dummy_488) * np.sin(_Dummy_492) * np.sin(_Dummy_493) * np.cos(_Dummy_487) - np.sin(_Dummy_487) * np.sin(
                      _Dummy_492) * np.sin(_Dummy_493) * np.cos(_Dummy_488) + np.sin(_Dummy_488) * np.sin(_Dummy_493) * np.cos(
                      _Dummy_483) * np.cos(_Dummy_492) + np.cos(_Dummy_487) * np.cos(_Dummy_488) * np.cos(_Dummy_493)) / (L * m),
              _Dummy_494 * (-np.sin(_Dummy_483) * np.sin(_Dummy_488) * np.cos(_Dummy_487) * np.cos(_Dummy_492) + np.sin(
                  _Dummy_487) * np.cos(_Dummy_488) * np.cos(_Dummy_492) + np.sin(_Dummy_488) * np.sin(_Dummy_492) * np.cos(
                  _Dummy_483)) / (L * m * np.cos(_Dummy_493))], [0, 0, 0, -_Dummy_494 * (
                        np.sin(_Dummy_483) * np.cos(_Dummy_492) * np.cos(_Dummy_493) + np.sin(_Dummy_487) * np.sin(_Dummy_493) * np.cos(
                    _Dummy_483) - np.sin(_Dummy_492) * np.cos(_Dummy_483) * np.cos(_Dummy_487) * np.cos(_Dummy_493)) * np.sin(
                _Dummy_492) * np.cos(_Dummy_488) * np.cos(_Dummy_493) / (m + mL), _Dummy_494 * (
                                                                          np.sin(_Dummy_483) * np.cos(_Dummy_492) * np.cos(
                                                                      _Dummy_493) + np.sin(_Dummy_487) * np.sin(
                                                                      _Dummy_493) * np.cos(_Dummy_483) - np.sin(
                                                                      _Dummy_492) * np.cos(_Dummy_483) * np.cos(
                                                                      _Dummy_487) * np.cos(_Dummy_493)) * np.sin(
                _Dummy_493) * np.cos(_Dummy_488) / (m + mL), -_Dummy_494 * (np.sin(_Dummy_483) * np.cos(_Dummy_492) * np.cos(
                _Dummy_493) + np.sin(_Dummy_487) * np.sin(_Dummy_493) * np.cos(_Dummy_483) - np.sin(_Dummy_492) * np.cos(
                _Dummy_483) * np.cos(_Dummy_487) * np.cos(_Dummy_493)) * np.cos(_Dummy_488) * np.cos(_Dummy_492) * np.cos(
                _Dummy_493) / (m + mL), (_Dummy_489 * np.cos(_Dummy_488) + _Dummy_490 * np.sin(_Dummy_488)) / np.cos(
                _Dummy_483) ** 2, 0, (_Dummy_489 * np.cos(_Dummy_488) + _Dummy_490 * np.sin(_Dummy_488)) * np.sin(
                _Dummy_483) / np.cos(_Dummy_483) ** 2, 0, 0, 0, 0, 0, _Dummy_494 * (
                                                                          np.sin(_Dummy_483) * np.sin(_Dummy_493) * np.cos(
                                                                      _Dummy_492) - np.sin(_Dummy_487) * np.cos(
                                                                      _Dummy_483) * np.cos(_Dummy_493) - np.sin(
                                                                      _Dummy_492) * np.sin(_Dummy_493) * np.cos(
                                                                      _Dummy_483) * np.cos(_Dummy_487)) * np.cos(
                _Dummy_488) / (L * m), _Dummy_494 * (np.sin(_Dummy_483) * np.sin(_Dummy_492) + np.cos(_Dummy_483) * np.cos(
                _Dummy_487) * np.cos(_Dummy_492)) * np.cos(_Dummy_488) / (L * m * np.cos(_Dummy_493))], [0, 0, 0, -(
                        -_Dummy_494 * (-np.sin(_Dummy_483) * np.sin(_Dummy_487) * np.cos(_Dummy_488) + np.sin(_Dummy_488) * np.cos(
                    _Dummy_487)) * np.sin(_Dummy_492) * np.cos(_Dummy_493) + _Dummy_494 * (
                                    np.sin(_Dummy_483) * np.cos(_Dummy_487) * np.cos(_Dummy_488) + np.sin(_Dummy_487) * np.sin(
                                _Dummy_488)) * np.sin(_Dummy_493)) * np.sin(_Dummy_492) * np.cos(_Dummy_493) / (m + mL), (
                                                                                                            -_Dummy_494 * (
                                                                                                                -np.sin(
                                                                                                                    _Dummy_483) * np.sin(
                                                                                                            _Dummy_487) * np.cos(
                                                                                                            _Dummy_488) + np.sin(
                                                                                                            _Dummy_488) * np.cos(
                                                                                                            _Dummy_487)) * np.sin(
                                                                                                        _Dummy_492) * np.cos(
                                                                                                        _Dummy_493) + _Dummy_494 * (
                                                                                                                        np.sin(_Dummy_483) * np.cos(
                                                                                                                    _Dummy_487) * np.cos(
                                                                                                                    _Dummy_488) + np.sin(
                                                                                                                    _Dummy_487) * np.sin(
                                                                                                                    _Dummy_488)) * np.sin(
                                                                                                        _Dummy_493)) * np.sin(
                _Dummy_493) / (m + mL), -(-_Dummy_494 * (
                        -np.sin(_Dummy_483) * np.sin(_Dummy_487) * np.cos(_Dummy_488) + np.sin(_Dummy_488) * np.cos(_Dummy_487)) * np.sin(
                _Dummy_492) * np.cos(_Dummy_493) + _Dummy_494 * (np.sin(_Dummy_483) * np.cos(_Dummy_487) * np.cos(_Dummy_488) + np.sin(
                _Dummy_487) * np.sin(_Dummy_488)) * np.sin(_Dummy_493)) * np.cos(_Dummy_492) * np.cos(_Dummy_493) / (m + mL), 0, 0,
                                                                                                0, 0, 0, 0, 0, 0,
                                                                                                _Dummy_494 * (
                                                                                                            np.sin(_Dummy_483) * np.sin(
                                                                                                        _Dummy_487) * np.sin(
                                                                                                        _Dummy_492) * np.sin(
                                                                                                        _Dummy_493) * np.cos(
                                                                                                        _Dummy_488) - np.sin(
                                                                                                        _Dummy_483) * np.cos(
                                                                                                        _Dummy_487) * np.cos(
                                                                                                        _Dummy_488) * np.cos(
                                                                                                        _Dummy_493) - np.sin(
                                                                                                        _Dummy_487) * np.sin(
                                                                                                        _Dummy_488) * np.cos(
                                                                                                        _Dummy_493) - np.sin(
                                                                                                        _Dummy_488) * np.sin(
                                                                                                        _Dummy_492) * np.sin(
                                                                                                        _Dummy_493) * np.cos(
                                                                                                        _Dummy_487)) / (
                                                                                                            L * m),
                                                                                                _Dummy_494 * (-np.sin(
                                                                                                    _Dummy_483) * np.sin(
                                                                                                    _Dummy_487) * np.cos(
                                                                                                    _Dummy_488) + np.sin(
                                                                                                    _Dummy_488) * np.cos(
                                                                                                    _Dummy_487)) * np.cos(
                                                                                                    _Dummy_492) / (
                                                                                                            L * m * np.cos(
                                                                                                        _Dummy_493))],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, (-_Dummy_489 * Jx + _Dummy_489 * Jz) / Jy,
              (_Dummy_490 * Jx - _Dummy_490 * Jy) / Jz, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, np.sin(_Dummy_488) * np.tan(_Dummy_483), np.cos(_Dummy_488), np.sin(_Dummy_488) / np.cos(_Dummy_483),
              (_Dummy_489 * Jy - _Dummy_489 * Jz) / Jx, 0, (_Dummy_491 * Jx - _Dummy_491 * Jy) / Jz, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, np.cos(_Dummy_488) * np.tan(_Dummy_483), -np.sin(_Dummy_488), np.cos(_Dummy_488) / np.cos(_Dummy_483),
              (_Dummy_490 * Jy - _Dummy_490 * Jz) / Jx, (-_Dummy_491 * Jx + _Dummy_491 * Jz) / Jy, 0, 0, 0, 0, 0],
             [0, 0, 0, (-3 * _Dummy_477 ** 2 * L * mL * np.sin(_Dummy_493) * np.cos(
                 _Dummy_493) ** 2 - _Dummy_478 ** 2 * L * mL * np.sin(_Dummy_493) - 2 * _Dummy_494 * np.sin(_Dummy_483) * np.sin(
                 _Dummy_487) * np.cos(_Dummy_488) * np.cos(_Dummy_493) ** 2 + _Dummy_494 * np.sin(_Dummy_483) * np.sin(
                 _Dummy_487) * np.cos(_Dummy_488) - 2 * _Dummy_494 * np.sin(_Dummy_483) * np.sin(_Dummy_492) * np.sin(
                 _Dummy_493) * np.cos(_Dummy_487) * np.cos(_Dummy_488) * np.cos(_Dummy_493) - 2 * _Dummy_494 * np.sin(
                 _Dummy_487) * np.sin(_Dummy_488) * np.sin(_Dummy_492) * np.sin(_Dummy_493) * np.cos(
                 _Dummy_493) + 2 * _Dummy_494 * np.sin(_Dummy_488) * np.cos(_Dummy_487) * np.cos(
                 _Dummy_493) ** 2 - _Dummy_494 * np.sin(_Dummy_488) * np.cos(_Dummy_487) - 2 * _Dummy_494 * np.sin(
                 _Dummy_493) * np.cos(_Dummy_483) * np.cos(_Dummy_488) * np.cos(_Dummy_492) * np.cos(_Dummy_493)) * np.sin(
                 _Dummy_492) / (m + mL), (
                          -3 * _Dummy_477 ** 2 * L * mL * np.cos(_Dummy_493) ** 3 + 2 * _Dummy_477 ** 2 * L * mL * np.cos(
                      _Dummy_493) - _Dummy_478 ** 2 * L * mL * np.cos(_Dummy_493) + 2 * _Dummy_494 * np.sin(_Dummy_483) * np.sin(
                      _Dummy_487) * np.sin(_Dummy_493) * np.cos(_Dummy_488) * np.cos(_Dummy_493) - 2 * _Dummy_494 * np.sin(
                      _Dummy_483) * np.sin(_Dummy_492) * np.cos(_Dummy_487) * np.cos(_Dummy_488) * np.cos(
                      _Dummy_493) ** 2 + _Dummy_494 * np.sin(_Dummy_483) * np.sin(_Dummy_492) * np.cos(_Dummy_487) * np.cos(
                      _Dummy_488) - 2 * _Dummy_494 * np.sin(_Dummy_487) * np.sin(_Dummy_488) * np.sin(_Dummy_492) * np.cos(
                      _Dummy_493) ** 2 + _Dummy_494 * np.sin(_Dummy_487) * np.sin(_Dummy_488) * np.sin(
                      _Dummy_492) - 2 * _Dummy_494 * np.sin(_Dummy_488) * np.sin(_Dummy_493) * np.cos(_Dummy_487) * np.cos(
                      _Dummy_493) - 2 * _Dummy_494 * np.cos(_Dummy_483) * np.cos(_Dummy_488) * np.cos(_Dummy_492) * np.cos(
                      _Dummy_493) ** 2 + _Dummy_494 * np.cos(_Dummy_483) * np.cos(_Dummy_488) * np.cos(_Dummy_492)) / (m + mL), (
                          -3 * _Dummy_477 ** 2 * L * mL * np.sin(_Dummy_493) * np.cos(
                      _Dummy_493) ** 2 - _Dummy_478 ** 2 * L * mL * np.sin(_Dummy_493) - 2 * _Dummy_494 * np.sin(
                      _Dummy_483) * np.sin(_Dummy_487) * np.cos(_Dummy_488) * np.cos(_Dummy_493) ** 2 + _Dummy_494 * np.sin(
                      _Dummy_483) * np.sin(_Dummy_487) * np.cos(_Dummy_488) - 2 * _Dummy_494 * np.sin(_Dummy_483) * np.sin(
                      _Dummy_492) * np.sin(_Dummy_493) * np.cos(_Dummy_487) * np.cos(_Dummy_488) * np.cos(
                      _Dummy_493) - 2 * _Dummy_494 * np.sin(_Dummy_487) * np.sin(_Dummy_488) * np.sin(_Dummy_492) * np.sin(
                      _Dummy_493) * np.cos(_Dummy_493) + 2 * _Dummy_494 * np.sin(_Dummy_488) * np.cos(_Dummy_487) * np.cos(
                      _Dummy_493) ** 2 - _Dummy_494 * np.sin(_Dummy_488) * np.cos(_Dummy_487) - 2 * _Dummy_494 * np.sin(
                      _Dummy_493) * np.cos(_Dummy_483) * np.cos(_Dummy_488) * np.cos(_Dummy_492) * np.cos(_Dummy_493)) * np.cos(
                 _Dummy_492) / (m + mL), 0, 0, 0, 0, 0, 0, 0, 0, _Dummy_494 * (
                          np.sin(_Dummy_483) * np.sin(_Dummy_487) * np.sin(_Dummy_493) * np.cos(_Dummy_488) - np.sin(_Dummy_483) * np.sin(
                      _Dummy_492) * np.cos(_Dummy_487) * np.cos(_Dummy_488) * np.cos(_Dummy_493) - np.sin(_Dummy_487) * np.sin(
                      _Dummy_488) * np.sin(_Dummy_492) * np.cos(_Dummy_493) - np.sin(_Dummy_488) * np.sin(_Dummy_493) * np.cos(
                      _Dummy_487) - np.cos(_Dummy_483) * np.cos(_Dummy_488) * np.cos(_Dummy_492) * np.cos(_Dummy_493)) / (L * m),
              _Dummy_494 * (
                          np.sin(_Dummy_483) * np.cos(_Dummy_487) * np.cos(_Dummy_488) * np.cos(_Dummy_492) + np.sin(_Dummy_487) * np.sin(
                      _Dummy_488) * np.cos(_Dummy_492) - np.sin(_Dummy_492) * np.cos(_Dummy_483) * np.cos(_Dummy_488)) * np.sin(
                  _Dummy_493) / (L * m * np.cos(_Dummy_493) ** 2)], [0, 0, 0, (
                        _Dummy_477 ** 2 * L * mL * np.cos(_Dummy_492) * np.cos(
                    _Dummy_493) ** 2 + _Dummy_478 ** 2 * L * mL * np.cos(_Dummy_492) - _Dummy_494 * np.sin(_Dummy_483) * np.sin(
                    _Dummy_487) * np.sin(_Dummy_493) * np.cos(_Dummy_488) * np.cos(_Dummy_492) + 2 * _Dummy_494 * np.sin(
                    _Dummy_483) * np.sin(_Dummy_492) * np.cos(_Dummy_487) * np.cos(_Dummy_488) * np.cos(_Dummy_492) * np.cos(
                    _Dummy_493) + 2 * _Dummy_494 * np.sin(_Dummy_487) * np.sin(_Dummy_488) * np.sin(_Dummy_492) * np.cos(
                    _Dummy_492) * np.cos(_Dummy_493) + _Dummy_494 * np.sin(_Dummy_488) * np.sin(_Dummy_493) * np.cos(
                    _Dummy_487) * np.cos(_Dummy_492) + 2 * _Dummy_494 * np.cos(_Dummy_483) * np.cos(_Dummy_488) * np.cos(
                    _Dummy_492) ** 2 * np.cos(_Dummy_493) - _Dummy_494 * np.cos(_Dummy_483) * np.cos(_Dummy_488) * np.cos(
                    _Dummy_493)) * np.cos(_Dummy_493) / (m + mL), _Dummy_494 * (-np.sin(_Dummy_483) * np.cos(_Dummy_487) * np.cos(
                _Dummy_488) * np.cos(_Dummy_492) - np.sin(_Dummy_487) * np.sin(_Dummy_488) * np.cos(_Dummy_492) + np.sin(
                _Dummy_492) * np.cos(_Dummy_483) * np.cos(_Dummy_488)) * np.sin(_Dummy_493) * np.cos(_Dummy_493) / (m + mL),
                                                                  -_Dummy_494 * (
                                                                              -np.sin(_Dummy_483) * np.cos(_Dummy_487) * np.cos(
                                                                          _Dummy_488) * np.cos(_Dummy_492) - np.sin(
                                                                          _Dummy_487) * np.sin(_Dummy_488) * np.cos(
                                                                          _Dummy_492) + np.sin(_Dummy_492) * np.cos(
                                                                          _Dummy_483) * np.cos(_Dummy_488)) * np.cos(
                                                                      _Dummy_492) * np.cos(_Dummy_493) ** 2 / (m + mL) + (
                                                                              _Dummy_494 * (np.sin(_Dummy_483) * np.sin(
                                                                          _Dummy_487) * np.cos(_Dummy_488) - np.sin(
                                                                          _Dummy_488) * np.cos(_Dummy_487)) * np.sin(
                                                                          _Dummy_493) - _Dummy_494 * (
                                                                                          np.sin(_Dummy_483) * np.cos(
                                                                                      _Dummy_487) * np.cos(
                                                                                      _Dummy_488) + np.sin(
                                                                                      _Dummy_487) * np.sin(
                                                                                      _Dummy_488)) * np.sin(
                                                                          _Dummy_492) * np.cos(
                                                                          _Dummy_493) - _Dummy_494 * np.cos(
                                                                          _Dummy_483) * np.cos(_Dummy_488) * np.cos(
                                                                          _Dummy_492) * np.cos(_Dummy_493) - L * mL * (
                                                                                          _Dummy_477 ** 2 * np.cos(
                                                                                      _Dummy_493) ** 2 + _Dummy_478 ** 2)) * np.sin(
                                                                      _Dummy_492) * np.cos(_Dummy_493) / (m + mL), 0, 0, 0,
                                                                  0, 0, 0, 0, 0, _Dummy_494 * (
                                                                              -np.sin(_Dummy_483) * np.cos(_Dummy_487) * np.cos(
                                                                          _Dummy_488) * np.cos(_Dummy_492) - np.sin(
                                                                          _Dummy_487) * np.sin(_Dummy_488) * np.cos(
                                                                          _Dummy_492) + np.sin(_Dummy_492) * np.cos(
                                                                          _Dummy_483) * np.cos(_Dummy_488)) * np.sin(
                    _Dummy_493) / (L * m), -_Dummy_494 * (np.sin(_Dummy_483) * np.sin(_Dummy_492) * np.cos(_Dummy_487) * np.cos(
                    _Dummy_488) + np.sin(_Dummy_487) * np.sin(_Dummy_488) * np.sin(_Dummy_492) + np.cos(_Dummy_483) * np.cos(
                    _Dummy_488) * np.cos(_Dummy_492)) / (L * m * np.cos(_Dummy_493))],
             [0, 0, 0, 2 * _Dummy_478 * L * mL * np.sin(_Dummy_492) * np.cos(_Dummy_493) / (m + mL),
              -2 * _Dummy_478 * L * mL * np.sin(_Dummy_493) / (m + mL),
              2 * _Dummy_478 * L * mL * np.cos(_Dummy_492) * np.cos(_Dummy_493) / (m + mL), 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 2 * _Dummy_477 * L * mL * np.sin(_Dummy_492) * np.cos(_Dummy_493) ** 3 / (m + mL),
              -2 * _Dummy_477 * L * mL * np.sin(_Dummy_493) * np.cos(_Dummy_493) ** 2 / (m + mL),
              2 * _Dummy_477 * L * mL * np.cos(_Dummy_492) * np.cos(_Dummy_493) ** 3 / (m + mL), 0, 0, 0, 0, 0, 0, 0, 1, 0,
              0]])

    def funcB(self, _Dummy_504, _Dummy_503, _Dummy_502, _Dummy_498, _Dummy_497, _Dummy_496, _Dummy_510, _Dummy_505,
              _Dummy_509, _Dummy_513, _Dummy_512, _Dummy_511, _Dummy_515, _Dummy_514, _Dummy_500, _Dummy_499,
              _Dummy_516, _Dummy_508, _Dummy_507, _Dummy_506, m, mL, g, L, Jx, Jy, Jz, _Dummy_517, _Dummy_501):
        return np.array([[0, 0, 0, -(
                    (np.sin(_Dummy_505) * np.sin(_Dummy_509) * np.cos(_Dummy_510) - np.sin(_Dummy_510) * np.cos(_Dummy_509)) * np.sin(
                _Dummy_515) - (np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(_Dummy_509) * np.sin(
                _Dummy_510)) * np.sin(_Dummy_514) * np.cos(_Dummy_515) - np.cos(_Dummy_505) * np.cos(_Dummy_510) * np.cos(
                _Dummy_514) * np.cos(_Dummy_515)) * np.sin(_Dummy_514) * np.cos(_Dummy_515) / (m + mL), ((np.sin(_Dummy_505) * np.sin(
            _Dummy_509) * np.cos(_Dummy_510) - np.sin(_Dummy_510) * np.cos(_Dummy_509)) * np.sin(_Dummy_515) - (
                                                                                                            np.sin(_Dummy_505) * np.cos(
                                                                                                        _Dummy_509) * np.cos(
                                                                                                        _Dummy_510) + np.sin(
                                                                                                        _Dummy_509) * np.sin(
                                                                                                        _Dummy_510)) * np.sin(
            _Dummy_514) * np.cos(_Dummy_515) - np.cos(_Dummy_505) * np.cos(_Dummy_510) * np.cos(_Dummy_514) * np.cos(
            _Dummy_515)) * np.sin(_Dummy_515) / (m + mL), -(
                    (np.sin(_Dummy_505) * np.sin(_Dummy_509) * np.cos(_Dummy_510) - np.sin(_Dummy_510) * np.cos(_Dummy_509)) * np.sin(
                _Dummy_515) - (np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(_Dummy_509) * np.sin(
                _Dummy_510)) * np.sin(_Dummy_514) * np.cos(_Dummy_515) - np.cos(_Dummy_505) * np.cos(_Dummy_510) * np.cos(
                _Dummy_514) * np.cos(_Dummy_515)) * np.cos(_Dummy_514) * np.cos(_Dummy_515) / (m + mL), 0, 0, 0, 0, 0, 0, 0, 0, (
                                      (-(np.sin(_Dummy_505) * np.sin(_Dummy_509) * np.cos(_Dummy_510) - np.sin(_Dummy_510) * np.cos(
                                          _Dummy_509)) * np.sin(_Dummy_514) * np.cos(_Dummy_515) - (
                                                   np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(
                                               _Dummy_509) * np.sin(_Dummy_510)) * np.sin(_Dummy_515)) * np.sin(_Dummy_515) + (-(
                                          np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(_Dummy_509) * np.sin(
                                      _Dummy_510)) * np.cos(_Dummy_514) * np.cos(_Dummy_515) + np.sin(_Dummy_514) * np.cos(
                                  _Dummy_505) * np.cos(_Dummy_510) * np.cos(_Dummy_515)) * np.cos(_Dummy_514) * np.cos(
                                  _Dummy_515)) * np.sin(_Dummy_514) * np.sin(_Dummy_515) / (L * m) + (-(
                    (np.sin(_Dummy_505) * np.sin(_Dummy_509) * np.cos(_Dummy_510) - np.sin(_Dummy_510) * np.cos(_Dummy_509)) * np.cos(
                _Dummy_514) * np.cos(_Dummy_515) + np.sin(_Dummy_515) * np.cos(_Dummy_505) * np.cos(_Dummy_510)) * np.sin(
            _Dummy_515) - (-(
                    np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(_Dummy_509) * np.sin(_Dummy_510)) * np.cos(
            _Dummy_514) * np.cos(_Dummy_515) + np.sin(_Dummy_514) * np.cos(_Dummy_505) * np.cos(_Dummy_510) * np.cos(
            _Dummy_515)) * np.sin(_Dummy_514) * np.cos(_Dummy_515)) * np.sin(_Dummy_515) * np.cos(_Dummy_514) / (L * m) + ((-(
                    np.sin(_Dummy_505) * np.sin(_Dummy_509) * np.cos(_Dummy_510) - np.sin(_Dummy_510) * np.cos(_Dummy_509)) * np.sin(
            _Dummy_514) * np.cos(_Dummy_515) - (np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(
            _Dummy_509) * np.sin(_Dummy_510)) * np.sin(_Dummy_515)) * np.sin(_Dummy_514) * np.cos(_Dummy_515) - ((
                                                                                                                 np.sin(_Dummy_505) * np.sin(
                                                                                                             _Dummy_509) * np.cos(
                                                                                                             _Dummy_510) - np.sin(
                                                                                                             _Dummy_510) * np.cos(
                                                                                                             _Dummy_509)) * np.cos(
            _Dummy_514) * np.cos(_Dummy_515) + np.sin(_Dummy_515) * np.cos(_Dummy_505) * np.cos(_Dummy_510)) * np.cos(
            _Dummy_514) * np.cos(_Dummy_515)) * np.cos(_Dummy_515) / (L * m), -((-(
                    np.sin(_Dummy_505) * np.sin(_Dummy_509) * np.cos(_Dummy_510) - np.sin(_Dummy_510) * np.cos(_Dummy_509)) * np.sin(
            _Dummy_514) * np.cos(_Dummy_515) - (np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(
            _Dummy_509) * np.sin(_Dummy_510)) * np.sin(_Dummy_515)) * np.sin(_Dummy_515) + (-(
                    np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(_Dummy_509) * np.sin(_Dummy_510)) * np.cos(
            _Dummy_514) * np.cos(_Dummy_515) + np.sin(_Dummy_514) * np.cos(_Dummy_505) * np.cos(_Dummy_510) * np.cos(
            _Dummy_515)) * np.cos(_Dummy_514) * np.cos(_Dummy_515)) * np.cos(_Dummy_514) * np.cos(_Dummy_515) / (L * m * (
                    np.sin(_Dummy_514) ** 2 * np.cos(_Dummy_515) ** 2 + np.cos(_Dummy_514) ** 2 * np.cos(_Dummy_515) ** 2)) + (-(
                    (np.sin(_Dummy_505) * np.sin(_Dummy_509) * np.cos(_Dummy_510) - np.sin(_Dummy_510) * np.cos(_Dummy_509)) * np.cos(
                _Dummy_514) * np.cos(_Dummy_515) + np.sin(_Dummy_515) * np.cos(_Dummy_505) * np.cos(_Dummy_510)) * np.sin(
            _Dummy_515) - (-(
                    np.sin(_Dummy_505) * np.cos(_Dummy_509) * np.cos(_Dummy_510) + np.sin(_Dummy_509) * np.sin(_Dummy_510)) * np.cos(
            _Dummy_514) * np.cos(_Dummy_515) + np.sin(_Dummy_514) * np.cos(_Dummy_505) * np.cos(_Dummy_510) * np.cos(
            _Dummy_515)) * np.sin(_Dummy_514) * np.cos(_Dummy_515)) * np.sin(_Dummy_514) * np.cos(_Dummy_515) / (L * m * (
                    np.sin(_Dummy_514) ** 2 * np.cos(_Dummy_515) ** 2 + np.cos(_Dummy_514) ** 2 * np.cos(_Dummy_515) ** 2))],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, Jx ** (-1.0), 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Jy ** (-1.0), 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Jz ** (-1.0), 0, 0, 0, 0]])

    def compute_control(self, state, setpoint, time):
        cur_load_pos = state["load_pos"]
        cur_load_vel = state["load_vel"]
        cur_pole_eul = state["pole_eul"]
        cur_pole_ang_vel = state["pole_ang_vel"]
        cur_quat = state["quat"]
        cur_ang_vel = state["ang_vel"]
        target_load_pos = setpoint["target_pos"]
        target_load_vel = setpoint["target_vel"]
        target_eul = setpoint["target_eul"]
        target_ang_vel = setpoint["target_ang_vel"]
        target_pole_eul = setpoint["target_pole_eul"]
        target_pole_ang_vel = setpoint["target_pole_ang_vel"]
        cur_quat = np.roll(cur_quat, -1)
        cur_eul = si.spatial.transform.Rotation.from_quat(cur_quat).as_euler('xyz')
        state = np.hstack((cur_load_pos, cur_load_vel, cur_eul, cur_ang_vel, cur_pole_eul, cur_pole_ang_vel))
        target = np.hstack((target_load_pos, target_load_vel, target_eul, target_ang_vel, target_pole_eul,
                            target_pole_ang_vel))
        if self.K is None or self.K is not None and self.controller_step > self.K.shape[0] - 1:
            ctrl = -self.K_lti @ (state - target)
        else:
            ctrl = -self.K[self.controller_step, :, :] @ (state - target)
        ctrl[0] = (self.mass + self.payload_mass) * self.gravity + ctrl[0]
        return ctrl

    def linearize_sys(self):
        m, mL, g, L, Jx, Jy, Jz, t = sp.symbols('m mL g L Jx Jy Jz t')
        x, y, z, phi, theta, psi, alpha, beta, omx, omy, omz, F, taux, tauy, tauz = sp.Function('x')(t), \
        sp.Function('y')(t), sp.Function('z')(t), sp.Function('phi')(t), sp.Function('theta')(t), sp.Function('psi')(t), \
        sp.Function('alpha')(t), sp.Function('beta')(t), sp.Function('omx')(t), sp.Function('omy')(t), \
        sp.Function('omz')(t), sp.Function('F')(t), sp.Function('taux')(t), sp.Function('tauy')(t), sp.Function('tauz')(t)

        dx, dy, dz = sp.diff(x, t), sp.diff(y, t), sp.diff(z, t)
        dalpha, dbeta = sp.diff(alpha, t), sp.diff(beta, t)
        ddalpha, ddbeta = sp.diff(dalpha, t), sp.diff(dbeta, t)
        xi = sp.Matrix([x, y, z, dx, dy, dz, phi, theta, psi, omx, omy, omz, alpha, beta, dalpha, dbeta])
        u = sp.Matrix([F, taux, tauy, tauz])

        S_phi = sp.sin(phi)
        S_theta = sp.sin(theta)
        S_psi = sp.sin(psi)
        C_phi = sp.cos(phi)
        C_theta = sp.cos(theta)
        C_psi = sp.cos(psi)

        W = sp.Matrix([[1, 0, -S_theta], [0, C_phi, C_theta * S_phi], [0, -S_phi, C_theta * C_phi]])
        W_inv = W.inv()

        R = (sp.Matrix([[1, 0, 0], [0, C_phi, S_phi], [0, -S_phi, C_phi]]) *
             sp.Matrix([[C_theta, 0, -S_theta], [0, 1, 0], [S_theta, 0, C_theta]]) *
             sp.Matrix([[C_psi, S_psi, 0], [-S_psi, C_psi, 0], [0, 0, 1]])).T

        S_alpha = sp.sin(alpha)
        S_beta = sp.sin(beta)
        C_alpha = sp.cos(alpha)
        C_beta = sp.cos(beta)
        R_q = (sp.Matrix([[1, 0, 0], [0, C_alpha, S_alpha], [0, -S_alpha, C_alpha]]) *
               sp.Matrix([[C_beta, 0, -S_beta], [0, 1, 0], [S_beta, 0, C_beta]])).T
        q = R_q * sp.Matrix([0, 0, -1])
        dq = sp.diff(q, t)
        ddq = sp.diff(dq, t)

        H = sp.Matrix([[ddq[0].coeff(ddalpha), ddq[1].coeff(ddalpha), ddq[2].coeff(ddalpha)],
                       [ddq[0].coeff(ddbeta), ddq[1].coeff(ddbeta), ddq[2].coeff(ddbeta)]]).T

        H_inv = (H.T * H).inv() * H.T
        # print(sp.latex(H_inv))
        # print(sp.latex(ddq))
        # print(ddq[1].coeff(ddbeta))

        om = sp.Matrix([omx, omy, omz])
        e3 = sp.Matrix([0, 0, 1])

        f1 = 1 / (m + mL) * (q.dot(F*R*e3) - mL*L*dq.dot(dq))*q - sp.Matrix([0, 0, g])
        f2 = sp.Matrix([[1/Jx, 0, 0],[0, 1/Jy, 0],[0, 0, 1/Jz]]) * (sp.Matrix([taux, tauy, tauz]) -
                                                                    om.cross(sp.Matrix([Jx*omx, Jy*omy, Jz*omz])))
        f3 = H_inv * (1 / (m*L) * q.cross(q.cross(F*R*e3)) - dq.dot(dq)*q)

        f = sp.zeros(16, 1)
        f[0:3, :] = sp.Matrix([dx, dy, dz])
        f[3:6, :] = f1
        f[6:9, :] = W_inv * om
        f[9:12, :] = f2
        f[12:14, :] = sp.Matrix([dalpha, dbeta])
        f[14:16, :] = f3

        f = sp.simplify(sp.trigsimp(f))

        A = sp.diff(f, xi)
        B = sp.diff(f, u)
        A = A[:, 0, :, 0]
        B = B[:, 0, :, 0]
        print(sp.latex(sp.simplify(sp.trigsimp(f))))
        # print(sp.latex(B))
        state_input = [elem[0] for elem in sp.Matrix.vstack(xi, u).tolist()]
        param_sym = [m, mL, g, L, Jx, Jy, Jz, sp.Derivative(0.0, t), sp.Derivative(1, t)]
        self.A_lambda = sp.lambdify(state_input + param_sym, A)
        self.B_lambda = sp.lambdify(state_input + param_sym, B)

    def compute_lti_lqr(self, states=np.zeros(16), inputs=np.zeros(4)):
        xi0 = states
        u0 = inputs
        u0[0] = (self.mass + self.payload_mass)*self.gravity
        setpoint = np.hstack((xi0, u0)).tolist()
        param_num = [self.mass, self.payload_mass, self.gravity, self.L, self.inertia[0], self.inertia[1],
                     self.inertia[2], 0, 0]
        A = self.funcA(*(setpoint + param_num)).T
        B = self.funcB(*(setpoint + param_num)).T
        # A = self.A_lambda(*(setpoint + param_num)).T
        # B = self.B_lambda(*(setpoint + param_num)).T

        expM = np.vstack((np.hstack((A, B)), np.zeros((4, 20)))) * self.dt
        # print(expM)
        blkmtx = si.linalg.expm(expM)
        A_d = blkmtx[0:16, 0:16]
        B_d = blkmtx[0:16, 16:]
        self.K_lti, _, _ = control.dlqr(A_d, B_d, self.Q_d, self.R_d, method='scipy')
        # self.K, _, _ = control.lqr(A, B, self.Q, self.R, method='scipy')


class LtvLqrLoadControl(LqrLoadControl):
    def __init__(self, mass, inertia, gravity):
        super().__init__(mass, inertia, gravity)
        self.compute_lti_lqr()

    def compute_ltv_lqr(self, target_state, target_input, payload_mass, dt):
        N = target_input.shape[0]
        # compute state and input matrix
        setpoints = np.hstack((target_state, target_input))

        A = np.asarray([np.eye(16) + dt * self.funcA(*(setpoints[i, :].tolist() + [self.mass] + [payload_mass[i]] +
                                                 [self.gravity, self.L, self.inertia[0],
                     self.inertia[1], self.inertia[2], 0, 0])).T for i in range(N)])
        B = np.asarray([dt * self.funcB(*(setpoints[i, :].tolist() + [self.mass] + [payload_mass[i]] +
                                                 [self.gravity, self.L, self.inertia[0],
                     self.inertia[1], self.inertia[2], 0, 0])).T for i in range(N)])

        self.K = np.zeros((N, target_input.shape[1], target_state.shape[1]))
        P = np.zeros((N, target_state.shape[1], target_state.shape[1]))

        # Backward pass
        for i in range(2, N+1):
            self.K[N-i, :, :] = np.linalg.inv(self.R_d + B[N-i, :, :].T @ P[N-i+1, :, :] @ B[N-i, :, :]) @ \
                           B[N-i, :, :].T @ P[N-i+1, :, :] @ A[N-i, :, :]
            P[N-i, :, :] = self.Q_d + self.K[N-i, :, :].T @ self.R_d @ self.K[N-i, :, :] + \
                           (A[N-i, :, :] - B[N-i, :, :] @ self.K[N-i, :, :]).T @ P[N-i+1, :, :] @ \
                           (A[N-i, :, :] - B[N-i, :, :] @ self.K[N-i, :, :])