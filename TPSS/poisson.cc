
/*
 * test poisson problem
 *
 *  Created on: Sep 12, 2019
 *      Author: witte
 */

#include "poisson.h"
#include "ct_parameter.h"

using namespace dealii;
using namespace Laplace;

struct Timings
{
  std::vector<Utilities::MPI::MinMaxAvg> setup;
  std::vector<Utilities::MPI::MinMaxAvg> apply;
};

struct TestParameter
{
  TPSS::PatchVariant    patch_variant       = CT::PATCH_VARIANT_;
  TPSS::SmootherVariant smoother_variant    = CT::SMOOTHER_VARIANT_;
  double                cg_reduction        = 1.e-8;
  unsigned              n_refinements       = 1;
  unsigned              n_smoothing_steps   = 1;
  unsigned              n_samples           = 10;
  unsigned              n_subsamples_vmult  = 100;
  unsigned              n_subsamples_smooth = 20;
  unsigned              n_subsamples_mg     = 10;

  std::string
  to_string() const
  {
    std::ostringstream oss;
    oss << Util::parameter_to_fstring("n_samples:", n_samples);
    oss << Util::parameter_to_fstring("n_subsamples_vmult:", n_subsamples_vmult);
    oss << Util::parameter_to_fstring("n_subsamples_smooth:", n_subsamples_smooth);
    oss << Util::parameter_to_fstring("n_subsamples_mg:", n_subsamples_mg);
    oss << Util::parameter_to_fstring("solver reduction:", cg_reduction);
    oss << std::endl;
    return oss.str();
  }
};

template<int dim, int fe_degree>
struct Test
{
  static constexpr int n_patch_dofs_per_direction =
    TPSS::UniversalInfo<dim>::n_cells_per_direction(CT::PATCH_VARIANT_) * (fe_degree + 1);
  using PoissonProblem =
    typename Poisson::ModelProblem<dim, fe_degree, double, n_patch_dofs_per_direction>;
  using VECTOR           = typename PoissonProblem::VECTOR;
  using SCHWARZ_SMOOTHER = typename PoissonProblem::SCHWARZ_SMOOTHER;
  using SYSTEM_MATRIX    = typename PoissonProblem::SYSTEM_MATRIX;
  using LEVEL_MATRIX     = typename PoissonProblem::LEVEL_MATRIX;

  const TestParameter prms;
  Laplace::Parameter  parameters;
  RT::Parameter       rt_parameters;

  Test(const TestParameter & prms_in = TestParameter{}) : prms(prms_in)
  {
    {
      //: discretization
      parameters.n_cycles  = 1;
      parameters.n_refines = prms.n_refinements;
      //: iterative solver
      parameters.solver_reduction      = prms.cg_reduction;
      parameters.solver_max_iterations = 100;
      parameters.precondition_variant  = Parameter::PreconditionVariant::MG;
      //: multigrid
      parameters.coarse_level                           = 1;
      parameters.schwarz_smoother_data.patch_variant    = prms.patch_variant;
      parameters.schwarz_smoother_data.smoother_variant = prms.smoother_variant;
      parameters.schwarz_smoother_data.manual_coloring  = true;
      const double damping_factor =
        TPSS::lookup_damping_factor(prms.patch_variant, prms.smoother_variant, dim);
      parameters.schwarz_smoother_data.damping_factor            = damping_factor;
      parameters.schwarz_smoother_data.number_of_smoothing_steps = prms.n_smoothing_steps;
      parameters.mg_smoother_post_reversed                       = true;
    }

    //: discretization
    rt_parameters.n_cycles              = 1;
    rt_parameters.mesh.geometry_variant = MeshParameter::GeometryVariant::Cube;
    rt_parameters.mesh.n_refinements    = 1;
    rt_parameters.mesh.n_repetitions    = 2;
    //: iterative solver
    rt_parameters.solver.rel_tolerance        = prms.cg_reduction;
    rt_parameters.solver.precondition_variant = SolverParameter::PreconditionVariant::GMG;
    //: multigrid
    rt_parameters.multigrid.coarse_level     = 1;
    rt_parameters.multigrid.smoother.variant = SmootherParameter::SmootherVariant::Schwarz;
    rt_parameters.multigrid.smoother.schwarz.patch_variant    = prms.patch_variant;
    rt_parameters.multigrid.smoother.schwarz.smoother_variant = prms.smoother_variant;
    rt_parameters.multigrid.smoother.schwarz.manual_coloring  = true;
    const double damping_factor =
      TPSS::lookup_damping_factor(prms.patch_variant, prms.smoother_variant, dim);
    rt_parameters.multigrid.smoother.schwarz.damping_factor = damping_factor;
    rt_parameters.multigrid.smoother.n_smoothing_steps      = prms.n_smoothing_steps;
    rt_parameters.multigrid.mg_smoother_post_reversed       = true;
  }

  std::string
  write_header()
  {
    std::ostringstream oss;
    oss << Util::git_version_to_fstring();

    oss << Util::parameter_to_fstring("Date:", Utilities::System::get_date());
    oss << Util::parameter_to_fstring("Number of MPI processes:",
                                      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
    oss << Util::parameter_to_fstring("Vectorization level:",
                                      Utilities::System::get_current_vectorization_level());
    oss << std::endl;

    oss << prms.to_string();
    return oss.str();
  }

  std::string
  write_ppdata_to_string(const Laplace::PostProcessData & pp_data)
  {
    std::ostringstream oss;
    ConvergenceTable   info_table;
    Assert(!pp_data.n_cells_global.empty(), ExcMessage("No cells to post process."));
    for(unsigned run = 0; run < pp_data.n_cells_global.size(); ++run)
    {
      info_table.add_value("n_levels", pp_data.n_mg_levels.at(run));
      info_table.add_value("n_cells", pp_data.n_cells_global.at(run));
      info_table.add_value("n_dofs", pp_data.n_dofs_global.at(run));
      info_table.add_value("n_colors", pp_data.n_colors_system.at(run));
      info_table.add_value("n_iter", pp_data.n_iterations_system.at(run));
      info_table.add_value("reduction", pp_data.average_reduction_system.at(run));
    }

    info_table.set_scientific("reduction", true);
    info_table.set_precision("reduction", 3);
    info_table.write_text(oss);

    return oss.str();
  }

  std::string
  write_timings_to_string(const Timings & timings, const Laplace::PostProcessData & pp_data)
  {
    std::ostringstream oss;
    ConvergenceTable   timings_table;
    for(unsigned n = 0; n < timings.apply.size(); ++n)
    {
      timings_table.add_value("sample", n + 1);
      timings_table.add_value("procs", Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
      timings_table.add_value("dofs", pp_data.n_dofs_global.at(0));
      timings_table.add_value("setup (max)", timings.setup[n].max);
      timings_table.add_value("setup (min)", timings.setup[n].min);
      timings_table.add_value("setup (avg)", timings.setup[n].avg);
      timings_table.add_value("apply (max)", timings.apply[n].max);
      timings_table.add_value("apply (min)", timings.apply[n].min);
      timings_table.add_value("apply (avg)", timings.apply[n].avg);
    }

    timings_table.set_scientific("setup (max)", true);
    timings_table.set_scientific("setup (min)", true);
    timings_table.set_scientific("setup (avg)", true);
    timings_table.set_scientific("apply (max)", true);
    timings_table.set_scientific("apply (min)", true);
    timings_table.set_scientific("apply (avg)", true);
    timings_table.write_text(oss, TableHandler::TextOutputFormat::org_mode_table);

    return oss.str();
  }

  //: determine filename from test parameters
  std::string
  get_filename(const PoissonProblem & poisson_problem)
  {
    std::ostringstream oss;
    const auto         n_dofs_global = poisson_problem.pp_data.n_dofs_global.front();
    const auto         n_mpi_procs   = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

    oss << TPSS::getstr_schwarz_variant(CT::PATCH_VARIANT_, CT::SMOOTHER_VARIANT_);
    oss << "_" << dim << "D";
    oss << "_" << fe_degree << "deg";
    oss << "_" << n_mpi_procs << "procs";
    oss << "_" << Util::si_metric_prefix(n_dofs_global) << "DoFs";
    return oss.str();
  }

  /**
   * test a single operator application, smoothing step and v-cycle
   */
  void
  partial()
  {
    Timings        timings_vmult, timings_smooth, timings_mg, timings_total;
    PoissonProblem poisson_problem{rt_parameters, parameters};
    poisson_problem.create_triangulation(parameters.n_refines);
    poisson_problem.distribute_dofs();

    for(unsigned sample = 0; sample < prms.n_samples; ++sample)
    {
      *(poisson_problem.pcout) << "Compute sample " << sample << std::endl;
      Timer time(MPI_COMM_WORLD, true);

      //: setup (total)
      time.restart();
      poisson_problem.prepare_linear_system();
      const auto & gmg_preconditioner = poisson_problem.prepare_preconditioner_mg();
      time.stop();
      timings_total.setup.push_back(time.get_last_lap_wall_time_data());

      //: solve (total)
      time.restart();
      poisson_problem.solve(gmg_preconditioner);
      time.stop();
      timings_total.apply.push_back(time.get_last_lap_wall_time_data());

      {
        Timer time(MPI_COMM_WORLD, true);

        //: setup
        time.restart();
        const auto    mf_storage = poisson_problem.template build_mf_storage<double>();
        SYSTEM_MATRIX system_matrix;
        system_matrix.initialize(mf_storage);
        time.stop();
        timings_vmult.setup.push_back(time.get_last_lap_wall_time_data());

        //: vmult
        VECTOR tmp;
        mf_storage->initialize_dof_vector(tmp);
        fill_with_random_values(tmp);
        VECTOR dst = tmp;
        time.restart();
        for(unsigned subsample = 0; subsample < prms.n_subsamples_vmult; ++subsample)
          system_matrix.vmult(dst, tmp);
        time.stop();
        Utilities::MPI::MinMaxAvg t_apply = time.get_last_lap_wall_time_data();
        t_apply                           = t_apply / prms.n_subsamples_vmult;
        timings_vmult.apply.push_back(t_apply);
      }

      {
        Timer time(MPI_COMM_WORLD, true);

        //: setup
        time.restart();
        const unsigned fine_level = poisson_problem.triangulation.n_global_levels() - 1;
        const auto     mf_storage = poisson_problem.template build_mf_storage<double>(fine_level);
        LEVEL_MATRIX   level_matrix;
        level_matrix.initialize(mf_storage);
        const auto subdomain_handler = poisson_problem.build_patch_storage(fine_level, mf_storage);
        const auto & schwarz_data    = poisson_problem.parameters.schwarz_smoother_data;
        const auto   schwarz_preconditioner =
          poisson_problem.build_schwarz_preconditioner(subdomain_handler,
                                                       level_matrix,
                                                       schwarz_data);
        typename SCHWARZ_SMOOTHER::AdditionalData smoother_data;
        smoother_data.number_of_smoothing_steps = schwarz_data.number_of_smoothing_steps;
        SCHWARZ_SMOOTHER schwarz_smoother;
        schwarz_smoother.initialize(level_matrix, schwarz_preconditioner, smoother_data);
        time.stop();
        timings_smooth.setup.push_back(time.get_last_lap_wall_time_data());

        //: smooth
        VECTOR tmp;
        mf_storage->initialize_dof_vector(tmp);
        fill_with_random_values(tmp);
        VECTOR dst = tmp;
        time.restart();
        for(unsigned subsample = 0; subsample < prms.n_subsamples_smooth; ++subsample)
          schwarz_smoother.step(dst, tmp);
        time.stop();
        Utilities::MPI::MinMaxAvg t_apply = time.get_last_lap_wall_time_data();
        t_apply                           = t_apply / prms.n_subsamples_smooth;
        timings_smooth.apply.push_back(t_apply);
      }

      {
        Timer time(MPI_COMM_WORLD, true);

        //: setup
        time.restart();
        const auto & gmg_preconditioner = poisson_problem.prepare_preconditioner_mg();
        time.stop();
        timings_mg.setup.push_back(time.get_last_lap_wall_time_data());

        //: V-cycle
        VECTOR tmp;
        poisson_problem.system_matrix.get_matrix_free()->initialize_dof_vector(tmp);
        fill_with_random_values(tmp);
        VECTOR dst = tmp;
        time.restart();
        for(unsigned subsample = 0; subsample < prms.n_subsamples_mg; ++subsample)
          gmg_preconditioner.vmult(dst, tmp);
        time.stop();
        Utilities::MPI::MinMaxAvg t_apply = time.get_last_lap_wall_time_data();
        t_apply                           = t_apply / prms.n_subsamples_mg;
        timings_mg.apply.push_back(t_apply);
      }
    } // end sample loop


    //: write performance timings
    const std::string filename = get_filename(poisson_problem);
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::fstream                     fstream;
      const Laplace::PostProcessData & pp_data = poisson_problem.pp_data;

      fstream.open("vmult_" + filename + ".time", std::ios_base::out);
      fstream << write_timings_to_string(timings_vmult, pp_data);
      fstream.close();

      fstream.open("smooth_" + filename + ".time", std::ios_base::out);
      fstream << write_timings_to_string(timings_smooth, pp_data);
      fstream.close();

      fstream.open("mg_" + filename + ".time", std::ios_base::out);
      fstream << write_timings_to_string(timings_mg, pp_data);
      fstream.close();

      fstream.open("solve_" + filename + ".time", std::ios_base::out);
      fstream << write_timings_to_string(timings_total, pp_data);
      fstream.close();
    }
  }

  /**
   * run complete Poisson problem once writing a generic log-file
   */
  void
  full()
  {
    PoissonProblem     poisson_problem{rt_parameters, parameters};
    std::ostringstream oss;
    const bool         is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    poisson_problem.pcout            = std::make_shared<ConditionalOStream>(oss, is_first_proc);
    oss << write_header();

    poisson_problem.run();
    const auto pp_data = poisson_problem.pp_data;
    oss << write_ppdata_to_string(pp_data);

    if(is_first_proc)
    {
      const std::string filename = get_filename(poisson_problem);
      std::fstream      fstream;
      fstream.open("poisson_" + filename + ".log", std::ios_base::out);
      fstream << oss.str();
      fstream.close();
    }
  }
};


int
main(int argc, char * argv[])
{
  // deallog.depth_console(3);

  // *** init TBB and MPI
  constexpr unsigned int           max_threads = 1; // no multithreading !?
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, max_threads);

  // *** static parameters
  constexpr unsigned dim       = CT::DIMENSION_;
  constexpr unsigned fe_degree = CT::FE_DEGREE_;

  // *** run-time parameters
  TestParameter prms;
  if(argc > 1)
    prms.n_refinements = std::atoi(argv[1]);
  if(argc > 2)
    prms.n_samples = std::atoi(argv[2]);

  // *** run tests
  Timer                time(MPI_COMM_WORLD, true);
  const bool           is_first_proc = (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  const auto           pcout         = ConditionalOStream(std::cout, is_first_proc);
  Test<dim, fe_degree> tester(prms);

  time.restart();
  tester.partial();
  time.stop();
  pcout << "total wall-time testing MF::vmult, TPSS::smooth, MG::vmult: " << std::endl;
  time.print_last_lap_wall_time_data(pcout);

  time.start();
  tester.full();
  time.stop();
  pcout << "total wall-time of PoissonProblem::run(): " << std::endl;
  time.print_last_lap_wall_time_data(pcout);
  pcout << "overall wall-time: " << std::endl;
  time.print_accumulated_wall_time_data(pcout);

  return 0;
}
