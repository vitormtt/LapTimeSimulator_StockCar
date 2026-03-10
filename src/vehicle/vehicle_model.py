import numpy as np


class VehicleModel:
    pass


class BicycleVehicle2DOF(VehicleModel):
    """
    Modelo de Bicicleta (2-DOF expandido para 3-DOF de Rolagem).
    Agora inclui transferência de carga lateral explícita com Roll Gradient.
    """

    def __init__(self, mass, wheelbase, a, cg_height, izz, engine_sys, brake_sys, trans_sys, tire_sys, track_width=2.5, k_roll=150000.0, roll_center_h=0.4, k_roll_front=None, k_roll_rear=None):
        self.mass = mass
        self.wheelbase = wheelbase
        self.a = a            # Distância CG ao eixo Dianteiro
        self.b = wheelbase - a  # Distância CG ao eixo Traseiro
        self.cg_height = cg_height
        self.izz = izz
        self.track_width = track_width

        # Parâmetros do 3º DOF (Rolagem - Roll)
        self.k_roll = k_roll           # Rigidez torcional global (Nm/rad)
        self.k_roll_front = k_roll_front if k_roll_front is not None else k_roll / 2.0
        self.k_roll_rear = k_roll_rear if k_roll_rear is not None else k_roll / 2.0
        self.roll_center_h = roll_center_h  # Altura do Centro de Rolagem (m)
        self.h_roll = self.cg_height - self.roll_center_h  # Braço de alavanca de rolagem

        # Subsistemas Modulares
        self.engine = engine_sys
        self.brakes = brake_sys
        self.transmission = trans_sys
        self.tires = tire_sys

        # Estados
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0
        self.roll_angle = 0.0  # Phi (rad)

    def calculate_derivatives(self, throttle, brake_pedal, steering_angle, current_rpm):
        """
        Calcula as forças de tração do trem de força.
        No modelo QSS espacial, as derivadas laterais são forçadas pelo raio,
        então aqui retornamos a capacidade limite motriz.
        """
        torque_motor = self.engine.get_max_torque(current_rpm) * throttle

        ratio = self.transmission.get_total_ratio(
            self.transmission.select_optimal_gear(
                self.vx, self.brakes.wheel_radius)
        )
        torque_roda = torque_motor * ratio * self.transmission.efficiency

        Fx_traction = torque_roda / self.brakes.wheel_radius

        # CORREÇÃO: O sistema de freios retorna um dicionário de forças ('front', 'rear', 'total')
        brake_forces = self.brakes.get_brake_force(brake_pedal, self.vx)
        Fx_brake = brake_forces['total']

        Fx_total = Fx_traction - Fx_brake

        return {
            'Fx_total': Fx_total,
            'torque_motor': torque_motor,
            'rpm': current_rpm
        }

    def calculate_roll_transfer(self, a_lat_g):
        """
        Equacionamento 3-DOF.
        Calcula o ângulo de rolagem da cabine/chassi e o Delta de Força Vertical 
        nos pneus externos em curvas (Lateral Weight Transfer).

        a_lat_g : Aceleração lateral em m/s²
        """
        # Força Centrífuga atuando no CG
        F_centrifuga = self.mass * a_lat_g

        # Momento de Rolagem (Roll Moment) no eixo geométrico de rolagem
        M_roll = F_centrifuga * self.h_roll

        # Ângulo de Rolagem Estacionário (Phi)
        self.roll_angle = M_roll / self.k_roll

        # Transferência de Carga Total do lado de Dentro para o lado de Fora (Delta Fz_lat)
        # Delta_Fz = (F_centrifuga * cg_height) / track_width
        delta_fz_lat = (F_centrifuga * self.cg_height) / self.track_width

        return {
            'roll_angle_deg': np.degrees(self.roll_angle),
            'delta_fz_lat': delta_fz_lat
        }
