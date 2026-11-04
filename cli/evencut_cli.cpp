// evencut command-line tool: reads weights, prints the optimal split as JSON (or text).
// Lets any language use the engine through a subprocess.

#include "evencut/evencut.hpp"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int exit_invalid = 1;
constexpr int exit_usage = 2;
constexpr int exit_internal = 3;

const char* const usage_text =
    "Usage: evencut -k N [options] [FILE]\n"
    "\n"
    "Splits an ordered list of non-negative weights into N contiguous parts as evenly as\n"
    "possible and prints the result as JSON. Weights are read from FILE or standard input,\n"
    "separated by whitespace, commas or newlines; a JSON array such as [5,3,8] also works and\n"
    "'#' starts a comment.\n"
    "\n"
    "Options:\n"
    "  -k, --parts N             number of parts (required)\n"
    "  -o, --objective NAME      min-max (default): smallest possible largest part\n"
    "                            sum-squares: smallest sum of squared part loads\n"
    "      --forbid B,C,...      never cut at position B, i.e. between items B-1 and B\n"
    "                            (numbered like the \"ends\" in the output; repeatable)\n"
    "      --secondary FILE      a second value per item, e.g. memory (min-max only)\n"
    "      --secondary-limit X   every part must keep its secondary sum <= X\n"
    "                            (required together with --secondary)\n"
    "      --no-polish           min-max: skip evening out the non-maximal parts\n"
    "      --scale X             read decimal weights and use round(weight * X)\n"
    "      --evaluate E1,E2,...  only measure the split with these end indices\n"
    "      --parents FILE        tree mode: parent index per node (-1 for the root);\n"
    "                            splits the tree into N connected parts (min-max)\n"
    "      --format json|text    output format (default: json)\n"
    "  -h, --help                show this help\n"
    "      --version             show the version\n"
    "\n"
    "Exit codes: 0 success, 1 invalid input, 2 usage error, 3 internal error.\n";

struct usage_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string read_all(const std::string& path) {
    if (path.empty() || path == "-") {
        std::ostringstream ss;
        ss << std::cin.rdbuf();
        return ss.str();
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) throw evencut::error("cannot open file: " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::vector<std::string> tokens_of(const std::string& text_in) {
    // Windows tools (PowerShell pipes, Notepad) often prepend a UTF-8 byte order mark.
    const std::string_view text = std::string_view(text_in).starts_with("\xEF\xBB\xBF")
                                      ? std::string_view(text_in).substr(3)
                                      : std::string_view(text_in);
    std::vector<std::string> out;
    std::string cur;
    bool comment = false;
    for (const char ch : text) {
        if (comment) {
            if (ch == '\n') comment = false;
            continue;
        }
        if (ch == '#') {
            comment = true;
        } else if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' || ch == ',' || ch == '[' || ch == ']') {
            if (!cur.empty()) out.push_back(std::move(cur));
            cur.clear();
            continue;
        } else {
            cur.push_back(ch);
            continue;
        }
        if (!cur.empty()) out.push_back(std::move(cur));
        cur.clear();
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

std::uint64_t parse_u64(std::string_view s, const std::string& what) {
    std::uint64_t v = 0;
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size()) {
        throw evencut::error(what + ": '" + std::string(s) + "' is not a non-negative integer that fits in 64 bits");
    }
    return v;
}

std::vector<std::uint64_t> parse_weights(const std::string& text, double scale, const std::string& what) {
    const std::vector<std::string> toks = tokens_of(text);
    if (scale <= 0.0) {
        std::vector<std::uint64_t> w;
        w.reserve(toks.size());
        for (const auto& t : toks) w.push_back(parse_u64(t, what));
        return w;
    }
    std::vector<double> values;
    values.reserve(toks.size());
    for (const auto& t : toks) {
        char* end = nullptr;
        const double v = std::strtod(t.c_str(), &end);
        if (end == t.c_str() || *end != '\0') throw evencut::error(what + ": '" + t + "' is not a number");
        values.push_back(v);
    }
    return evencut::quantize(values, scale);
}

std::vector<std::size_t> parse_index_list(const std::string& text, const std::string& what) {
    std::vector<std::size_t> out;
    for (const auto& t : tokens_of(text)) out.push_back(static_cast<std::size_t>(parse_u64(t, what)));
    return out;
}

std::vector<std::size_t> parse_parents(const std::string& text) {
    std::vector<std::size_t> out;
    for (const auto& t : tokens_of(text)) {
        if (t == "-1" || t == "root" || t == "null") {
            out.push_back(evencut::no_parent);
        } else {
            out.push_back(static_cast<std::size_t>(parse_u64(t, "parents")));
        }
    }
    return out;
}

template <class T>
void json_array(std::ostream& os, const std::vector<T>& v) {
    os << '[';
    for (std::size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    os << ']';
}

void print_partition(const evencut::partition_result& r, const std::string& objective, const std::string& format) {
    std::ostringstream os;
    os.precision(6);  // only `imbalance` is fractional; 6 significant digits read well
    if (format == "text") {
        os << "objective: " << objective << "\nparts: " << r.parts() << "\nmax_load: " << r.max_load
           << "\ntotal: " << r.total << "\nsum_of_squares: " << r.sum_of_squares << "\nimbalance: " << r.imbalance
           << "\n\npart\tbegin\tend\tload\n";
        for (std::size_t p = 0; p < r.parts(); ++p) {
            os << p << '\t' << r.part_begin(p) << '\t' << r.part_end(p) << '\t' << r.loads[p] << '\n';
        }
    } else {
        os << "{\"objective\":\"" << objective << "\",\"parts\":" << r.parts() << ",\"ends\":";
        json_array(os, r.ends);
        os << ",\"loads\":";
        json_array(os, r.loads);
        os << ",\"max_load\":" << r.max_load << ",\"total\":" << r.total << ",\"sum_of_squares\":\"" << r.sum_of_squares
           << "\",\"imbalance\":" << r.imbalance << "}\n";
    }
    std::cout << os.str();
}

void print_tree(const evencut::tree_result& r, const std::string& format) {
    std::ostringstream os;
    os.precision(6);  // only `imbalance` is fractional; 6 significant digits read well
    if (format == "text") {
        os << "objective: min-max (tree)\nparts: " << r.parts() << "\nmax_load: " << r.max_load << "\ntotal: " << r.total
           << "\nimbalance: " << r.imbalance << "\n\nnode\tpart\n";
        for (std::size_t v = 0; v < r.part_of.size(); ++v) os << v << '\t' << r.part_of[v] << '\n';
    } else {
        os << "{\"objective\":\"min-max\",\"parts\":" << r.parts() << ",\"part_of\":";
        json_array(os, r.part_of);
        os << ",\"loads\":";
        json_array(os, r.loads);
        os << ",\"cut_nodes\":";
        json_array(os, r.cut_nodes);
        os << ",\"max_load\":" << r.max_load << ",\"total\":" << r.total << ",\"sum_of_squares\":\"" << r.sum_of_squares
           << "\",\"imbalance\":" << r.imbalance << "}\n";
    }
    std::cout << os.str();
}

int run(int argc, char** argv) {
    std::size_t k = 0;
    bool have_k = false;
    std::string objective = "min-max";
    bool have_objective = false;
    std::string format = "json";
    std::string input_path;
    std::string secondary_path;
    std::string parents_path;
    std::string evaluate_list;
    bool have_evaluate = false;
    std::vector<std::size_t> forbidden;
    std::uint64_t secondary_limit = UINT64_MAX;
    bool have_limit = false;
    bool polish = true;
    double scale = 0.0;

    const auto value = [&](int& i, const char* name) -> std::string {
        if (i + 1 >= argc) throw usage_error(std::string("missing value for ") + name);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            std::cout << usage_text;
            return 0;
        } else if (a == "--version") {
            std::cout << "evencut " << evencut::version() << '\n';
            return 0;
        } else if (a == "-k" || a == "--parts") {
            k = static_cast<std::size_t>(parse_u64(value(i, "--parts"), "--parts"));
            have_k = true;
        } else if (a == "-o" || a == "--objective") {
            objective = value(i, "--objective");
            have_objective = true;
            if (objective != "min-max" && objective != "sum-squares") {
                throw usage_error("--objective must be min-max or sum-squares");
            }
        } else if (a == "--forbid") {
            const auto more = parse_index_list(value(i, "--forbid"), "--forbid");
            forbidden.insert(forbidden.end(), more.begin(), more.end());
        } else if (a == "--secondary") {
            secondary_path = value(i, "--secondary");
        } else if (a == "--secondary-limit") {
            secondary_limit = parse_u64(value(i, "--secondary-limit"), "--secondary-limit");
            have_limit = true;
        } else if (a == "--no-polish") {
            polish = false;
        } else if (a == "--scale") {
            const std::string s = value(i, "--scale");
            char* end = nullptr;
            scale = std::strtod(s.c_str(), &end);
            if (end == s.c_str() || *end != '\0' || !(scale > 0.0)) throw usage_error("--scale must be a positive number");
        } else if (a == "--evaluate") {
            evaluate_list = value(i, "--evaluate");
            have_evaluate = true;
        } else if (a == "--parents") {
            parents_path = value(i, "--parents");
        } else if (a == "--format") {
            format = value(i, "--format");
            if (format != "json" && format != "text") throw usage_error("--format must be json or text");
        } else if (!a.empty() && a[0] == '-' && a != "-") {
            throw usage_error("unknown option: " + a);
        } else if (input_path.empty()) {
            input_path = a;
        } else {
            throw usage_error("only one input file may be given");
        }
    }
    const bool tree_mode = !parents_path.empty();
    if (tree_mode && have_evaluate) throw usage_error("--parents and --evaluate cannot be combined");
    if (!have_k && !have_evaluate) throw usage_error("-k/--parts is required");
    if (have_limit && secondary_path.empty()) throw usage_error("--secondary-limit needs --secondary");
    if (!have_limit && !secondary_path.empty()) throw usage_error("--secondary needs --secondary-limit");
    // Options a mode cannot honour are errors, never silently ignored.
    if (tree_mode || have_evaluate) {
        const std::string mode = tree_mode ? "tree mode (--parents)" : "--evaluate";
        if (!forbidden.empty()) throw usage_error("--forbid is not supported in " + mode);
        if (!secondary_path.empty()) throw usage_error("--secondary is not supported in " + mode);
        if (!polish) throw usage_error("--no-polish is not supported in " + mode);
        if (have_objective && (have_evaluate || objective != "min-max")) {
            throw usage_error(tree_mode ? "tree mode supports only --objective min-max"
                                        : "--evaluate measures a given split; --objective does not apply");
        }
        if (have_evaluate && have_k) throw usage_error("--evaluate measures a given split; -k does not apply");
    }

    const std::vector<std::uint64_t> weights = parse_weights(read_all(input_path), scale, "weights");

    if (tree_mode) {
        const auto parents = parse_parents(read_all(parents_path));
        print_tree(evencut::partition_tree(parents, weights, k), format);
        return 0;
    }
    if (have_evaluate) {
        const auto ends = parse_index_list(evaluate_list, "--evaluate");
        print_partition(evencut::evaluate(weights, ends), "evaluate", format);
        return 0;
    }
    evencut::options opt;
    opt.goal = objective == "sum-squares" ? evencut::objective::sum_squares : evencut::objective::min_max;
    opt.forbidden_cuts = forbidden;
    opt.polish = polish;
    std::vector<std::uint64_t> secondary;
    if (!secondary_path.empty()) {
        secondary = parse_weights(read_all(secondary_path), 0.0, "secondary");
        opt.secondary = secondary;
        opt.secondary_limit = secondary_limit;  // required by the check above, so never unset here
    }
    print_partition(evencut::partition(weights, k, opt), objective, format);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const usage_error& e) {
        std::fprintf(stderr, "evencut: %s\nTry 'evencut --help'.\n", e.what());
        return exit_usage;
    } catch (const evencut::error& e) {
        const std::string msg = e.what();
        std::fprintf(stderr, "%s%s\n", msg.rfind("evencut:", 0) == 0 ? "" : "evencut: ", msg.c_str());
        return exit_invalid;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "evencut: internal error: %s\n", e.what());
        return exit_internal;
    }
}
