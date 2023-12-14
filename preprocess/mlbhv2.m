classdef mlbhv2 < handle
    properties (SetAccess = protected)
        filename
    end
    properties (Access = protected)
        fid
        var_pos  % [name start end]
        readonly
        indexed
        update_index
        fileinfo
    end
    
    methods
        function obj = mlbhv2(filename,mode)  % mode: read, write, append
            obj.fid = -1;
            if ~exist('mode','var'), mode = 'r'; end
            if ~exist('filename','var'), [n,p] = uigetfile( {'*.bhv2', 'BHV2 Files (*.bhv2)'} ); filename = [p n]; end
            obj.open(filename,mode);
        end
        function open(obj,filename,mode)
            close(obj);
            if ~exist('mode','var'), mode = 'r'; end
            obj.filename = filename;
            obj.readonly = false;
            obj.indexed = true;
            obj.update_index = false;
            obj.fileinfo = struct('machinefmt','ieee-le','encoding','UTF-8');
            switch lower(mode(1))
                case 'r'
                    obj.fid = fopen(filename,'r',obj.fileinfo.machinefmt,obj.fileinfo.encoding);
                    [~,info] = read_index(obj);
                    if ~strcmp(info.machinefmt,obj.fileinfo.machinefmt) || ~strcmp(info.encoding,obj.fileinfo.encoding)
                        obj.fileinfo = info; fclose(obj.fid); obj.fid = fopen(filename,'r',obj.fileinfo.machinefmt,obj.fileinfo.encoding);
                    end
                    obj.readonly = true;
                case 'w'
                    obj.fid = fopen(filename,'w',obj.fileinfo.machinefmt,obj.fileinfo.encoding);
                    write(obj,-1,'IndexPosition');
                    write(obj,obj.fileinfo,'FileInfo');
                case 'a'
                    if exist(filename,'file')
                        obj.fid = fopen(filename,'r+',obj.fileinfo.machinefmt,obj.fileinfo.encoding);
                        [pos,info] = read_index(obj);
                        if ~strcmp(info.machinefmt,obj.fileinfo.machinefmt) || ~strcmp(info.encoding,obj.fileinfo.encoding)
                            obj.fileinfo = info; fclose(obj.fid); obj.fid = fopen(filename,'r+',obj.fileinfo.machinefmt,obj.fileinfo.encoding);
                        end
                        if isempty(pos), fseek(obj.fid,0,1); else, fseek(obj.fid,pos,-1); end
                    else
                        obj.fid = fopen(filename,'w',obj.fileinfo.machinefmt,obj.fileinfo.encoding);
                        write(obj,-1,'IndexPosition');
                        write(obj,obj.fileinfo,'FileInfo');
                    end
            end
        end
        function close(obj)
            if -1~=obj.fid
                try
                    if obj.indexed && obj.update_index
                        s = ftell(obj.fid);
                        write_recursively(obj,obj.var_pos,'FileIndex');
                        fseek(obj.fid,0,-1);
                        write_recursively(obj,s,'IndexPosition');
                    end
                catch
                end
                fclose(obj.fid);
                obj.fid = -1;
            end
        end
        function val = isopen(obj), val = -1~=obj.fid; end
        function delete(obj), close(obj); end
        
        function s = write(obj,val,name)
            if obj.readonly, error('This file is not opened in the write or append mode.'); end
            if ~isempty(obj.var_pos) && ~isempty(find(strcmp(obj.var_pos(:,1),name),1)), error('The variable, %s, exists in the file already.',name); end
            idx = size(obj.var_pos,1);
            s = ftell(obj.fid);
            write_recursively(obj,val,name)
            idx = idx + 1;
            obj.var_pos{idx,1} = name;
            obj.var_pos{idx,2} = s;
            obj.var_pos{idx,3} = ftell(obj.fid);
            obj.update_index = true;
        end
        function val = read(obj,name)
            if ~obj.readonly, error('This file is not opened in the read mode.'); end
            pos = 0;
            if exist('name','var')
                if ~isempty(obj.var_pos)
                    row = find(strcmp(obj.var_pos(:,1),name),1);
                    if ~isempty(row)
                        fseek(obj.fid,obj.var_pos{row,2},-1);
                        val = read_variable(obj);
                        return
                    end
                    pos = obj.var_pos{end,3};
                end
            else
                obj.var_pos = [];
            end
            fseek(obj.fid,pos,-1);
            idx = size(obj.var_pos,1);
            while true
                try
                    s = ftell(obj.fid);
                    [a,b] = read_variable(obj);
                    idx = idx + 1;
                    obj.var_pos{idx,1} = b;
                    obj.var_pos{idx,2} = s;
                    obj.var_pos{idx,3} = ftell(obj.fid);
                    if exist('name','var')
                        if strcmp(b,name), val = a; return, end
                    else
                        val.(b) = a;
                    end
                catch err
                    if ~strcmp(err.identifier,'mlbhv2:eof'), rethrow(err); end
                    break;
                end
            end
            if ~exist('val','var'), val = []; end
        end
        function val = read_trial(obj)
            if ~obj.readonly, error('This file is not opened in the read mode.'); end
            if isempty(obj.var_pos)
                pos = 0;
            else
                for m=[obj.var_pos{~cellfun(@isempty,regexp(obj.var_pos(:,1),'^Trial\d+$','once')),2}]
                    fseek(obj.fid,m,-1);
                    [a,b] = read_variable(obj);
                    val(str2double(regexp(b,'\d+','match'))) = a; %#ok<AGROW>
                end
                pos = obj.var_pos{end,3};
            end
            fseek(obj.fid,pos,-1);
            idx = size(obj.var_pos,1);
            while true
                try
                    s = ftell(obj.fid);
                    [a,b] = read_variable(obj);
                    idx = idx + 1;
                    obj.var_pos{idx,1} = b;
                    obj.var_pos{idx,2} = s;
                    obj.var_pos{idx,3} = ftell(obj.fid);
                    if ~isempty(regexp(b,'^Trial\d+$','once')), val(str2double(regexp(b,'\d+','match'))) = a; end
                catch err
                    if ~strcmp(err.identifier,'mlbhv2:eof'), rethrow(err); end
                    break;
                end
            end
            if ~exist('val','var'), val = []; end
        end
        function val = who(obj)
            if obj.readonly
                if isempty(obj.var_pos), pos = 0; else pos = obj.var_pos{end,3}; end
                fseek(obj.fid,pos,-1);
                idx = size(obj.var_pos,1);
                while true
                    try
                        s = ftell(obj.fid);
                        [~,b] = read_variable(obj);
                        idx = idx + 1;
                        obj.var_pos{idx,1} = b;
                        obj.var_pos{idx,2} = s;
                        obj.var_pos{idx,3} = ftell(obj.fid);
                    catch err
                        if ~strcmp(err.identifier,'mlbhv2:eof'), rethrow(err); end
                        break;
                    end
                end
            end
            if isempty(obj.var_pos), val = []; else val = obj.var_pos(:,1); end
        end
    end
    
    methods (Access = protected)
        function [pos,fileinfo] = read_index(obj)
            obj.indexed = false;
            pos = [];
            fileinfo = struct('machinefmt','ieee-le','encoding','windows-1252');
            fseek(obj.fid,0,-1);
            lname = fread(obj.fid,1,'uint64=>double');
            name = fread(obj.fid,[1 lname],'char*1=>char');
            if strcmp(name,'IndexPosition')
                obj.indexed = true;
                fseek(obj.fid,0,-1);
                pos = read_variable(obj);

                s = ftell(obj.fid);
                lname = fread(obj.fid,1,'uint64=>double');
                name = fread(obj.fid,[1 lname],'char*1=>char');
                if strcmp(name,'FileInfo')
                    fseek(obj.fid,s,-1);
                    fileinfo = read_variable(obj);
                end
                
                if 0<pos
                    fseek(obj.fid,pos,-1);
                    [obj.var_pos,name] = read_variable(obj);
                    if ~strcmp(name,'FileIndex'), obj.var_pos = []; pos = []; end
                end
            end
        end
        function write_recursively(obj,val,name)
            type = class(val);
            if isobject(val), type = 'struct'; end
            switch type
                case 'struct'
                    dim = ndims(val);
                    sz = size(val);
                    field = fieldnames(val);
                    nfield = length(field);
                    fwrite(obj.fid,length(name),'uint64');
                    fwrite(obj.fid,name,'char*1');
                    fwrite(obj.fid,length(type),'uint64');
                    fwrite(obj.fid,type,'char*1');
                    fwrite(obj.fid,dim,'uint64');
                    fwrite(obj.fid,sz,'uint64');
                    fwrite(obj.fid,nfield,'uint64');
                    for m=1:prod(sz)
                        for n=1:nfield, write_recursively(obj,val(m).(field{n}),field{n}); end
                    end
                case 'cell'
                    dim = ndims(val);
                    sz = size(val);
                    fwrite(obj.fid,length(name),'uint64');
                    fwrite(obj.fid,name,'char*1');
                    fwrite(obj.fid,length(type),'uint64');
                    fwrite(obj.fid,type,'char*1');
                    fwrite(obj.fid,dim,'uint64');
                    fwrite(obj.fid,sz,'uint64');
                    for m=1:prod(sz), write_recursively(obj,val{m},''); end
                case 'function_handle'
                    write_variable(obj,name,func2str(val));
                otherwise
                    write_variable(obj,name,val);
            end
        end
        function write_variable(obj,name,val)
            dim = ndims(val);
            sz = size(val);
            type = class(val);
            fwrite(obj.fid,length(name),'uint64');
            fwrite(obj.fid,name,'char*1');
            fwrite(obj.fid,length(type),'uint64');
            fwrite(obj.fid,type,'char*1');
            fwrite(obj.fid,dim,'uint64');
            fwrite(obj.fid,sz,'uint64');
            fwrite(obj.fid,val,type);
        end
        function [val,name] = read_variable(obj)
            try
                lname = fread(obj.fid,1,'uint64=>double');
                if feof(obj.fid), error('mlbhv2:eof','End of file.'); end
                name = fread(obj.fid,[1 lname],'char*1=>char');
                ltype = fread(obj.fid,1,'uint64=>double');
                type = fread(obj.fid,[1 ltype],'char*1=>char');
                dim = fread(obj.fid,1,'uint64=>double');
                sz = fread(obj.fid,[1 dim],'uint64=>double');
                if strncmp(type,'ml',2), type = 'struct'; end
                switch type
                    case 'struct'
                        nfield = fread(obj.fid,1,'uint64=>double');
                        for m=1:prod(sz)
                            for n=1:nfield, [a,b] = read_variable(obj); val(m).(b) = a; end %#ok<AGROW>
                        end
                        if exist('val','var'), val = reshape(val,sz); else, val = []; end
                    case 'cell'
                        val = cell(sz);
                        for m=1:prod(sz), val{m} = read_variable(obj); end
                    otherwise
                        val = reshape(fread(obj.fid,prod(sz),['*' type]),sz);  % fread can handle only a 2-d size arg.
                end
            catch err
                rethrow(err);
            end
        end
    end
end
