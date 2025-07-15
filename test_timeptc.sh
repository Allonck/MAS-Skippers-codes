import argparse

def calculate_imaging_time(final_exposure_time, step_exposure_time, readout_time_per_image, images_per_step):
    """
    Calculates the total estimated time for an imaging sequence based on the provided formula.

    Args:
        final_exposure_time (float): The final exposure time in seconds.
        step_exposure_time (float): The step in exposure time between consecutive exposures in seconds.
        readout_time_per_image (float): The readout time per image in seconds.
        images_per_step (int): The number of images taken per distinct exposure step (M).

    Returns:
        float: The total estimated time in hours.

    Raises:
        ValueError: If images_per_step is not a positive integer, or if the final_exposure_time
                    is not a multiple of the step_exposure_time.
    """

    if not isinstance(images_per_step, int) or images_per_step <= 0:
        raise ValueError("The number of images per distinct exposure step (M) must be a positive integer.")
    if step_exposure_time <= 0 and final_exposure_time > 0:
        raise ValueError("The step in exposure time (S) must be a positive value if final_exposure_time is greater than 0.")
    if readout_time_per_image < 0:
        raise ValueError("The readout time per image (t_readout) cannot be negative.")

    # Calculate the number of distinct exposure steps (N/M)
    # The exposure times are 0, S, 2S, ..., final_exposure_time
    # This means the last step index 'i' will be final_exposure_time / step_exposure_time
    num_distinct_steps = 0
    if step_exposure_time > 0:
        try:
            num_distinct_steps = int(round(final_exposure_time / step_exposure_time))
        except ZeroDivisionError:
            pass # Already handled by step_exposure_time <= 0 check if final_exposure_time > 0

        # Validate that final_exposure_time is a multiple of step_exposure_time
        if not (abs(final_exposure_time % step_exposure_time) < 1e-9):
            raise ValueError("The final exposure time must be a multiple of the step in exposure time.")
    elif final_exposure_time > 0 and step_exposure_time == 0:
        raise ValueError("If step_exposure_time is 0, final_exposure_time must also be 0.")


    # Total number of images (N)
    # This includes one extra image at the beginning (at 0 exposure),
    # and then M images for each distinct exposure step from S up to final_exposure_time.
    # The number of distinct exposure steps from S to final_exposure_time is num_distinct_steps
    # (since the steps are S, 2S, ..., num_distinct_steps * S = final_exposure_time).
    N = 1 + (num_distinct_steps * images_per_step)

    # Calculate Total Readout Time (N * t_readout)
    total_readout_time = N * readout_time_per_image

    # Calculate Total Exposure Time (M * sum(i * S))
    # The exposures are 0, 0, 0, S, S, 2S, 2S, ... until final_exposure_time
    # The initial 0 exposure is one image.
    # The sum for total exposure time needs to account for the (M * sum(i * S)) part of the formula.
    # The sum goes from i=1 up to N/M. Here, N/M represents the number of groups of M images.
    # Since we have one initial image at 0, the remaining (N-1) images are distributed in (N-1)/M groups.
    # This means the sum is over the exposure steps from S, 2S, ..., (N-1)/M * S
    # So the counter 'i' for the sum goes from 1 to num_distinct_steps.
    sum_exposure_times = sum(i * step_exposure_time for i in range(1, num_distinct_steps + 1))
    total_exposure_time = images_per_step * sum_exposure_times

    # Total time in seconds
    total_time_seconds = total_readout_time + total_exposure_time

    # Convert to hours
    total_time_hours = total_time_seconds / 3600

    return total_time_hours

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcula el tiempo total estimado para una secuencia de imágenes con una exposición inicial de 0 y pasos incrementales.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-f", "--final_exposure",
        type=float,
        required=True,
        help="Tiempo de exposición final en segundos (por ejemplo, 60.0 para 1 minuto)."
    )
    parser.add_argument(
        "-s", "--step_exposure",
        type=float,
        required=True,
        help="Paso en el tiempo de exposición entre exposiciones consecutivas en segundos (por ejemplo, 5.0)."
    )
    parser.add_argument(
        "-r", "--readout_time",
        type=float,
        required=True,
        help="Tiempo de lectura por imagen en segundos (t_readout) (por ejemplo, 1.0)."
    )
    parser.add_argument(
        "-m", "--images_per_step",
        type=int,
        required=True,
        help="Número de imágenes tomadas por cada paso de exposición distinto (M) (por ejemplo, 3)."
    )

    args = parser.parse_args()

    try:
        estimated_time = calculate_imaging_time(
            args.final_exposure,
            args.step_exposure,
            args.readout_time,
            args.images_per_step
        )
        print(f"\n--- Resultado del Cálculo ---")
        print(f"Tiempo de Exposición Final (s): {args.final_exposure}")
        print(f"Paso en Tiempo de Exposición (s): {args.step_exposure}")
        print(f"Tiempo de Lectura por Imagen (s): {args.readout_time}")
        print(f"Imágenes por Paso de Exposición Distinto (M): {args.images_per_step}")
        print(f"Tiempo Total Estimado: {estimated_time:.4f} horas")
        print(f"Tiempo Total Estimado: {estimated_time * 60:.2f} minutos")
        print(f"Tiempo Total Estimado: {estimated_time * 3600:.2f} segundos\n")

    except ValueError as e:
        print(f"\n¡Error en los parámetros! 🚫 {e}\n")
    except Exception as e:
        print(f"\n¡Ocurrió un error inesperado! 😱 {e}\n")
