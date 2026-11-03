"""Standard-library source audit. Does NOT import or execute project modules."""
from pathlib import Path
import ast,hashlib,json,sys,warnings,symtable,builtins,re

ROOT=Path(__file__).resolve().parents[1]


def declarations(tree):
    names=set()
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):names.add(node.name)
        elif isinstance(node,ast.Assign):
            for target in node.targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n,ast.Name))
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):names.add(node.target.id)
        elif isinstance(node,(ast.Import,ast.ImportFrom)):
            for item in node.names:names.add(item.asname or item.name.split('.')[0])
    return names


def main():
    failures=[];trees={}
    files=sorted((ROOT/'src').rglob('*.py'))+sorted((ROOT/'tools').glob('*.py'))+[ROOT/'selftest.py']
    for path in files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',SyntaxWarning)
                source=path.read_text(encoding='utf-8');trees[path]=ast.parse(source,filename=str(path));compile(source,str(path),'exec')
        except (SyntaxError,SyntaxWarning) as exc:failures.append(f'{path}: {exc}')
    for path,tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node,ast.ImportFrom) or node.module is None:continue
            if node.level:
                base=path.parent
                for _ in range(node.level-1):base=base.parent
                target=base.joinpath(*node.module.split('.')).with_suffix('.py')
            else:target=(ROOT/'src').joinpath(*node.module.split('.')).with_suffix('.py')
            if target not in trees:continue
            exports=declarations(trees[target])
            for item in node.names:
                if item.name!='*' and item.name not in exports:
                    failures.append(f'{path.relative_to(ROOT)} imports missing {item.name} from {target.relative_to(ROOT)}')
    for path in files:
        source=path.read_text(encoding='utf-8')
        top=symtable.symtable(source,str(path),'exec')
        known=set(top.get_identifiers())|set(dir(builtins))|{'__file__','__name__','__package__','__class__'}
        def check_names(table):
            for symbol in table.get_symbols():
                if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in known:
                    failures.append(f'{path.relative_to(ROOT)}: unresolved global {symbol.get_name()} in {table.get_name()}')
            for child in table.get_children():check_names(child)
        check_names(top)
        for name in (re.findall(r'\b(?:[0-9]+[a-z]?_[A-Za-z_]+|benchmark)\.py\b',source) if path.is_relative_to(ROOT/'src') else []):
            if not (ROOT/'src'/name).exists():failures.append(f'{path.relative_to(ROOT)} references missing step {name}')
    manifest=json.loads((ROOT/'third_party'/'imuoptimize.manifest.json').read_text())
    for relative,digest in manifest['files_sha256'].items():
        path=ROOT/'third_party'/'imuoptimize'/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:failures.append(f'Vendor mismatch: {relative}')
    required=['0_check.py','1_synthesize.py','benchmark.py','2_upperbound.py','3_baseline.py','4_randoms.py',
              '5_greedy_a.py','6_greedy_b.py','7_swap.py','8_retrain.py','9_evaluation.py','10_analyze.py','11_visualizations.py']
    for name in required:
        if not (ROOT/'src'/name).is_file():failures.append(f'Missing step {name}')
    config_tree=trees[ROOT/'src'/'config.py']
    assignments={n.targets[0].id:n.value for n in config_tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    pool=ast.literal_eval(assignments['CANDIDATE_POOL'])
    if pool!=(0,1,2,4,5,7,8,9,15,16,17,18,19):failures.append('Unexpected reviewed pool.')
    mic=(ROOT/'src'/'shared'/'mic.py').read_text()
    if 'range(a,24)' in mic or 'range(24)' in mic:failures.append('MIC loops escaped the 13-site pool.')
    for obsolete in ('src/config_swap.py','src/swap_tools.py','src/imuplace','src/7b_swap_retrain_evaluate.py'):
        if (ROOT/obsolete).exists():failures.append(f'Obsolete add-on retained: {obsolete}')
    report={'status':'failed' if failures else 'passed','python_files_checked':len(files),
            'vendor_files_checked':len(manifest['files_sha256']),'candidate_pool':pool,
            'scope':'Syntax, local imported-name declarations, unresolved global names, step references, immutable vendor hashes and workflow inventory; no project execution.',
            'failures':failures}
    print(json.dumps(report,indent=2))
    return 1 if failures else 0


if __name__=='__main__':raise SystemExit(main())
